# education_search_full_education_with_longer_output.py
"""
Batch education lookup — full education entries (degree/institution/year).
Improvements:
- Increased max output tokens to reduce truncated replies.
- Smaller default batch size for safer debugging.
- Dumps raw responses to raw_responses/ for inspection.
- Robust retry/backoff and checkpointing.
"""

import os
import sys
import time
import re
import logging
import random
from dotenv import load_dotenv
import pandas as pd
import tenacity

# Try importing Google genai client (Gemini)
try:
    from google import genai
except Exception:
    genai = None

# ---------------- Configuration ----------------
INPUT_FILE = "input_data.xlsx"
OUTPUT_FILE = "output_with_education_ai.csv"
CHECKPOINT_FILE = "education_search_checkpoint.txt"
LOG_FILE = "education_search_full_education_long_output.log"

BATCH_SIZE = 10            # smaller while debugging; increase later
MODEL = "gemini-2.5-flash"
DAILY_LIMIT = 1000
SLEEP_SECONDS = 4

# increase output tokens to avoid truncation
MAX_OUTPUT_TOKENS = 2048  # increase if responses still truncated

NAME_COL = "Name"
ORG_COL = "Organization/Law Firm Name"
OUTPUT_COL = "Education (AI Found)"
YEAR_COL = "Education Year (AI Found)"

# Debug options
DUMP_RAW_RESPONSE = True   # will save raw responses to RAW_RESP_DIR for inspection
RAW_RESP_DIR = "raw_responses"

# ---------------- Logging ----------------
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)

# ---------------- Initialize client ----------------
load_dotenv()
client = None
if genai is not None:
    try:
        client = genai.Client()
        logging.info("Initialized genai client.")
    except Exception as e:
        logging.error("Failed to initialize genai client: %s", e, exc_info=True)
        client = None
else:
    logging.warning("google.genai import failed; please install google-genai SDK. client is None.")

# ---------------- Prompt: full education (strict) ----------------
PROMPT_TEMPLATE = """You are an expert researcher with access to legal and professional directories up to 2025.
Task: For each person below, find all education entries you can verify (degree + institution + year if available).
Rules (must follow exactly):
- Use only verifiable public sources (state bar, Martindale, law firm bio, official directories).
- Do NOT guess or invent years or degrees. If an item cannot be verified, write "N/A" for that element.
- List ALL education entries (e.g., B.S., B.A., J.D., LLM, PhD) and include degree name, institution and year if available.
- If multiple entries exist, separate them with a semicolon `;`.
- After each person's education line, append (source: <short URL or N/A>) — a single short source URL if available or N/A.
Required EXACT format:
1. <entry1; entry2; ...> (source: <url or N/A>)
Example:
1. B.S., Mechanical Engineering, Arizona State University, 1987; J.D., Pepperdine University, 1995 (source: https://example.com/profile)
If nothing is found, output exactly: N/A (source: N/A)
People to research:
"""

# ---------------- Utilities: checkpoint ----------------
def get_last_processed():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                s = f.read().strip()
                return int(s) if s else 0
        except Exception:
            return 0
    return 0

def save_checkpoint(idx):
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            f.write(str(idx))
    except Exception as e:
        logging.warning("Failed to save checkpoint: %s", e)

# ---------------- Utilities: extract and parse ----------------
def safe_get_response_text(response):
    if response is None:
        return ""
    try:
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text
    except Exception:
        pass
    for attr in ("content", "output", "answer", "response", "result"):
        try:
            val = getattr(response, attr, None)
            if isinstance(val, str):
                return val
            if isinstance(val, (list, tuple)):
                return "\n".join(map(str, val))
            if val is not None:
                return str(val)
        except Exception:
            continue
    return str(response)

def parse_numbered_list_full(text: str):
    parsed = {}
    if not text:
        return parsed

    lines = text.splitlines()
    current_key = None
    current_parts = []
    start_re = re.compile(r'^\s*(\d+)\s*[\.\)\-:]\s*(.*)$')
    alt_start_re = re.compile(r'^\s*(\d+)\s+(.+)$')

    for ln in lines:
        ln_strip = ln.rstrip()
        if not ln_strip:
            if current_key is not None:
                current_parts.append('')
            continue

        m = start_re.match(ln_strip)
        if m:
            if current_key is not None:
                parsed[current_key] = " ".join([p for p in current_parts if p is not None]).strip()
            current_key = m.group(1)
            rest = m.group(2).strip()
            current_parts = [rest] if rest else []
            continue

        m2 = alt_start_re.match(ln_strip)
        if m2 and current_key is None:
            if current_key is not None:
                parsed[current_key] = " ".join([p for p in current_parts if p is not None]).strip()
            current_key = m2.group(1)
            rest = m2.group(2).strip()
            current_parts = [rest] if rest else []
            continue

        if current_key is not None:
            current_parts.append(ln_strip)
        else:
            continue

    if current_key is not None:
        parsed[current_key] = " ".join([p for p in current_parts if p is not None]).strip()

    parsed = {k: (v.strip() if isinstance(v, str) else v) for k, v in parsed.items()}
    return parsed

def extract_jd_year_from_full_text(full_text: str):
    if not full_text or not full_text.strip():
        return "N/A"
    # attempt to find JD entry and 4-digit year in that entry
    entries = [e.strip() for e in re.split(r';\s*', full_text) if e.strip()]
    for entry in entries:
        if re.search(r'\bJ\.?D\.?\b', entry, flags=re.IGNORECASE):
            m = re.search(r'\b(19|20)\d{2}\b', entry)
            if m:
                return m.group(0)
            if re.search(r'year\s*[:\-]\s*n/?a', entry, flags=re.IGNORECASE):
                return "N/A"
    m_all = re.search(r'JD[^\d]{0,10}(\b(19|20)\d{2}\b)', full_text, flags=re.IGNORECASE)
    if m_all:
        return m_all.group(1)
    return "N/A"

# ---------------- Retry detection ----------------
def _is_retryable_exc(exc: Exception) -> bool:
    if exc is None:
        return False
    txt = str(exc).lower()
    if "503" in txt or "service unavailable" in txt or "unavailable" in txt:
        return True
    if "429" in txt or "rate limit" in txt or "quota" in txt:
        return True
    if "internal" in txt and "error" in txt:
        return True
    return False

# ---------------- Tenacity-wrapped low-level call ----------------
@tenacity.retry(
    retry=tenacity.retry_if_exception(lambda e: _is_retryable_exc(e)),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=120) + tenacity.wait_random(1, 6),
    stop=tenacity.stop_after_attempt(6),
    reraise=True,
)
def _call_api_with_retry(prompt: str, model: str = MODEL, temperature: float = 0.0):
    if client is None:
        raise RuntimeError("genai client not initialized.")
    try:
        # Use typed config and set max_output_tokens to avoid truncation
        try:
            config = genai.types.GenerateContentConfig(temperature=temperature, max_output_tokens=MAX_OUTPUT_TOKENS)
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception:
            # fallback without config
            resp = client.models.generate_content(model=model, contents=prompt)
        return resp
    except Exception as exc:
        try:
            headers = getattr(exc, "headers", None) or (getattr(exc, "response", None) and getattr(exc.response, "headers", None))
            if headers:
                ra = headers.get("Retry-After") or headers.get("retry-after")
                if ra:
                    try:
                        wait_seconds = int(ra)
                    except Exception:
                        wait_seconds = 30
                    jitter = random.uniform(0, 3)
                    logging.warning("Server requested Retry-After %s; sleeping %.1f s", ra, wait_seconds + jitter)
                    time.sleep(wait_seconds + jitter)
        except Exception:
            pass
        logging.error("Low-level API call raised exception: %s", exc, exc_info=True)
        raise

def call_api_safe(prompt: str):
    try:
        resp = _call_api_with_retry(prompt)
        txt = safe_get_response_text(resp)
        return True, txt
    except Exception as exc:
        logging.warning("Persistent API failure for prompt: %s", str(exc).splitlines()[0])
        return False, str(exc)

# ---------------- Main processing ----------------
def main():
    logging.info("Starting full-education fetch for %s (longer output)", INPUT_FILE)

    if not os.path.exists(INPUT_FILE):
        logging.error("Input file not found: %s", INPUT_FILE)
        return

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    total_rows = len(df)
    logging.info("Loaded %d rows", total_rows)

    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = "N/A"
    if YEAR_COL not in df.columns:
        df[YEAR_COL] = "N/A"

    start_row = get_last_processed()
    logging.info("Resuming from row (0-based): %d", start_row)

    if os.path.exists(OUTPUT_FILE) and start_row == 0:
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            if OUTPUT_COL in existing.columns and YEAR_COL in existing.columns and len(existing) <= len(df):
                df.loc[: len(existing) - 1, OUTPUT_COL] = existing.loc[: len(existing) - 1, OUTPUT_COL].values
                df.loc[: len(existing) - 1, YEAR_COL] = existing.loc[: len(existing) - 1, YEAR_COL].values
                start_row = len(existing)
                logging.info("Adjusted start_row to %d based on existing output.", start_row)
        except Exception as e:
            logging.warning("Could not load existing output file: %s", e)

    processed = start_row
    daily_requests = 0

    if DUMP_RAW_RESPONSE and not os.path.isdir(RAW_RESP_DIR):
        os.makedirs(RAW_RESP_DIR, exist_ok=True)

    try:
        for batch_start in range(start_row, total_rows, BATCH_SIZE):
            if daily_requests >= DAILY_LIMIT:
                logging.warning("Reached daily limit %d. Stopping.", DAILY_LIMIT)
                break

            batch_end = min(batch_start + BATCH_SIZE, total_rows)
            batch_df = df.iloc[batch_start:batch_end].copy()

            lines = []
            idx_map = []
            for j, (orig_index, row) in enumerate(batch_df.iterrows(), start=1):
                name = str(row.get(NAME_COL, "")).strip()
                org = str(row.get(ORG_COL, "")).strip()
                lines.append(f"{j}. {name} @ {org}")
                idx_map.append(orig_index)

            prompt = PROMPT_TEMPLATE + "\n".join(lines)

            logging.info("Processing rows %d..%d (size=%d)", batch_start, batch_end - 1, len(idx_map))
            print(f"\nProcessing rows {batch_start} to {batch_end - 1}")

            ok, response_text = call_api_safe(prompt)

            if ok:
                if DUMP_RAW_RESPONSE:
                    fname = os.path.join(RAW_RESP_DIR, f"resp_{batch_start}_{int(time.time())}.txt")
                    with open(fname, "w", encoding="utf-8") as fh:
                        fh.write(response_text)
                    logging.info("Wrote raw response to %s", fname)

                parsed = parse_numbered_list_full(response_text)

                for j, orig_idx in enumerate(idx_map, start=1):
                    key = str(j)
                    raw_val = parsed.get(key, "N/A")
                    if not raw_val or not raw_val.strip():
                        raw_val = "N/A"
                    # Expect format: "<entries> (source: <url or N/A>)"
                    # We'll keep full raw_val in OUTPUT_COL and try to extract JD year.
                    full_text = raw_val
                    jd_year = extract_jd_year_from_full_text(full_text)
                    df.at[orig_idx, OUTPUT_COL] = full_text
                    df.at[orig_idx, YEAR_COL] = jd_year
                    # show truncated preview to console
                    preview = (full_text[:200] + "...") if len(full_text) > 200 else full_text
                    print(f"  {df.at[orig_idx, NAME_COL][:40]:40} -> {preview} | Year: {jd_year}")

                processed = batch_end
                save_checkpoint(processed)
                daily_requests += 1
                logging.info("Completed batch up to %d. Checkpoint saved.", processed)

                try:
                    df.to_csv(OUTPUT_FILE, index=False)
                    logging.info("Intermediate save written to %s", OUTPUT_FILE)
                except Exception as e:
                    logging.warning("Failed intermediate save: %s", e)

                time.sleep(SLEEP_SECONDS + random.uniform(0, 1.0))

            else:
                logging.error("Persistent API failure for batch starting at %d. Marking rows N/A and continuing.", batch_start)
                for orig_idx in idx_map:
                    df.at[orig_idx, OUTPUT_COL] = "N/A"
                    df.at[orig_idx, YEAR_COL] = "N/A"
                processed = batch_end
                save_checkpoint(processed)
                try:
                    df.to_csv(OUTPUT_FILE, index=False)
                except Exception:
                    pass
                backoff = min(300, 30 + random.uniform(0, 30))
                logging.info("Backing off for %.1f s after persistent failure.", backoff)
                time.sleep(backoff)

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Saving progress up to %d and exiting.", processed)
        try:
            df.to_csv(OUTPUT_FILE, index=False)
        except Exception:
            pass
        print("\nInterrupted by user. Progress saved.")
        return

    try:
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info("Final output written to %s", OUTPUT_FILE)
    except Exception as e:
        logging.error("Failed to write final output: %s", e)

    logging.info("Job finished. Processed up to row %d of %d", processed, total_rows)
    print("\n--- Process Finished ---")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"Total rows processed: {processed}")

if __name__ == "__main__":
    main()
