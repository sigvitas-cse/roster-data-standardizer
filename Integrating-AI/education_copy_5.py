# education_search_with_years.py
"""
Batch education lookup (law school + JD year) using google.genai (Gemini).
- Output columns: "Education (AI Found)" and "Education Year (AI Found)".
- Robust retry/backoff, checkpointing, intermediate saves, optional raw-response dumps.
- Temperature=0.0 to reduce hallucination.
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

# Try importing the Google genai client (Gemini). If not installed, client will be None and script will raise.
try:
    from google import genai
except Exception:
    genai = None

# ---------------- Configuration ----------------
INPUT_FILE = "input_data.xlsx"                 # expects columns NAME_COL and ORG_COL
OUTPUT_FILE = "output_with_education_ai.csv"
CHECKPOINT_FILE = "education_search_checkpoint.txt"
LOG_FILE = "education_search_with_years.log"

BATCH_SIZE = 50            # set to 10 while debugging
MODEL = "gemini-2.5-flash"  # adjust if needed
DAILY_LIMIT = 1000
SLEEP_SECONDS = 5

NAME_COL = "Name"
ORG_COL = "Organization/Law Firm Name"
OUTPUT_COL = "Education (AI Found)"
YEAR_COL = "Education Year (AI Found)"

# Debugging: set to True to save raw model responses for first batches to tune parsing
DUMP_RAW_RESPONSE = False
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

# ---------------- Year-aware prompt (strict) ----------------
PROMPT_TEMPLATE = """You are an expert researcher with access to legal and professional directories up to 2025.
Task: For each person below, find the Law School (Juris Doctor, JD, LLM) and the JD graduation YEAR (if available).
Important rules (must follow exactly):
- Only use verifiable public/official sources (e.g. state bar records, Martindale, law firm bios, government directories). 
- Do NOT guess or invent years. If you cannot verify the year, return "N/A" for the year.
- Output ONLY the law school and the year in the EXACT format shown below.

Required output format (one line per person; numbered list):
1. <Law school name> (JD) — Year: <YYYY or N/A>
2. <Law school name> (LLM) — Year: <YYYY or N/A>
If multiple degrees are present, include both degrees but only list the JD year (or N/A).

People to research:
"""

# ---------------- Utility: checkpoint ----------------
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

# ---------------- Utilities: response extraction and parsing ----------------
def safe_get_response_text(response):
    """Best-effort extraction of text/string from SDK response object."""
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

def extract_school_and_year(value: str):
    """
    Normalize a single line value into (school_str, year_str).
    Accepts lines like:
      "Pepperdine University (JD) — Year: 1995"
      "University of San Diego (JD) — Year: N/A"
      "Harvard Law School (JD) 2001"
    Returns (school, year) with year as "YYYY" or "N/A".
    """
    if not value or not value.strip():
        return ("N/A", "N/A")
    v = value.strip()

    # 1) prefer explicit "Year: XXXX" (or "Year: N/A")
    m_year = re.search(r'year\s*[:\-]\s*(\d{4})', v, flags=re.IGNORECASE)
    if m_year:
        year = m_year.group(1)
        school = re.sub(r'[\-\—]\s*Year\s*[:\-]\s*\d{4}', '', v, flags=re.IGNORECASE).strip()
        return (school, year)

    m_na = re.search(r'year\s*[:\-]\s*(n/?a)', v, flags=re.IGNORECASE)
    if m_na:
        school = re.sub(r'[\-\—]\s*Year\s*[:\-]\s*(n/?a)', '', v, flags=re.IGNORECASE).strip()
        return (school, "N/A")

    # 2) detect 4-digit year anywhere (but be conservative)
    m4 = re.search(r'\b(19|20)\d{2}\b', v)
    if m4:
        year = m4.group(0)
        # Remove the found year from the string to keep school clean
        school = re.sub(r'\b' + re.escape(year) + r'\b', '', v).strip(' ,-–—;:')
        return (school, year)

    # 3) If there's no year, return school as-is and N/A (per strict rule)
    return (v, "N/A")

def parse_numbered_list_with_years(text: str):
    """
    Parse numbered outputs and return mapping:
      {'1': ('Pepperdine University (JD)', '1995'), ...}
    Accepts variations: "1. text", "1) text", "1 - text", etc.
    """
    parsed = {}
    if not text:
        return parsed

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # find first numbered line
    start = 0
    for i, ln in enumerate(lines):
        if re.match(r'^\s*\d+\s*[\.\)\-:]\s*', ln) or re.match(r'^\s*\d+\s+', ln):
            start = i
            break
    lines = lines[start:]

    pattern = re.compile(r'^\s*(\d+)\s*[\.\)\-:]\s*(.+)$')
    fallback = re.compile(r'^\s*(\d+)\s+(.+)$')

    for ln in lines:
        m = pattern.match(ln)
        if m:
            key = m.group(1)
            raw = m.group(2).strip()
        else:
            m2 = fallback.match(ln)
            if m2:
                key = m2.group(1)
                raw = m2.group(2).strip()
            else:
                continue
        school, year = extract_school_and_year(raw)
        parsed[key] = (school, year)
    return parsed

# ---------------- Retry detection ----------------
def _is_retryable_exc(exc: Exception) -> bool:
    if exc is None:
        return False
    txt = str(exc).lower()
    # common transient markers
    if "503" in txt or "service unavailable" in txt or "unavailable" in txt:
        return True
    if "429" in txt or "rate limit" in txt or "quota" in txt:
        return True
    if "internal" in txt and "error" in txt:
        return True
    return False

# ---------------- Tenacity-wrapped API call (low-level) ----------------
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
        # try typed config if SDK exposes it
        try:
            config = genai.types.GenerateContentConfig(temperature=temperature)
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception:
            resp = client.models.generate_content(model=model, contents=prompt)
        return resp
    except Exception as exc:
        # best-effort: honor Retry-After header if wrapped in exception (SDK-dependent)
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
                    logging.warning("Server asked to Retry-After %s s; sleeping %0.1f s", ra, wait_seconds + jitter)
                    time.sleep(wait_seconds + jitter)
        except Exception:
            pass
        logging.error("Low-level API call raised exception (transient?): %s", exc, exc_info=True)
        raise

def call_api_safe(prompt: str):
    """
    Safe wrapper around _call_api_with_retry:
    - returns (True, text) on success
    - returns (False, error_message) if retries exhausted / persistent failure
    """
    try:
        resp = _call_api_with_retry(prompt)
        txt = safe_get_response_text(resp)
        return True, txt
    except Exception as exc:
        logging.warning("Persistent API failure for prompt: %s", str(exc).splitlines()[0])
        return False, str(exc)

# ---------------- Main processing ----------------
def main():
    logging.info("Starting education fetch for %s", INPUT_FILE)

    if not os.path.exists(INPUT_FILE):
        logging.error("Input file not found: %s", INPUT_FILE)
        return

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    total_rows = len(df)
    logging.info("Loaded %d rows", total_rows)

    # Prepare columns
    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = "N/A"
    if YEAR_COL not in df.columns:
        df[YEAR_COL] = "N/A"

    start_row = get_last_processed()
    logging.info("Resuming from row (0-based): %d", start_row)

    # adopt existing output if present and starting at zero
    if os.path.exists(OUTPUT_FILE) and start_row == 0:
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            if OUTPUT_COL in existing.columns and YEAR_COL in existing.columns and len(existing) <= len(df):
                df.loc[: len(existing) - 1, OUTPUT_COL] = existing.loc[: len(existing) - 1, OUTPUT_COL].values
                df.loc[: len(existing) - 1, YEAR_COL] = existing.loc[: len(existing) - 1, YEAR_COL].values
                start_row = len(existing)
                logging.info("Adjusted start_row to %d based on existing output file.", start_row)
        except Exception as e:
            logging.warning("Could not load existing output file: %s", e)

    processed = start_row
    daily_requests = 0

    if DUMP_RAW_RESPONSE and not os.path.isdir(RAW_RESP_DIR):
        os.makedirs(RAW_RESP_DIR, exist_ok=True)

    try:
        for batch_start in range(start_row, total_rows, BATCH_SIZE):
            if daily_requests >= DAILY_LIMIT:
                logging.warning("Reached daily limit (%d). Stopping run.", DAILY_LIMIT)
                print("Reached DAILY_LIMIT. Stopping.")
                break

            batch_end = min(batch_start + BATCH_SIZE, total_rows)
            batch_df = df.iloc[batch_start:batch_end].copy()

            # Build prompt and index map
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
                    dump_file = os.path.join(RAW_RESP_DIR, f"resp_{batch_start}_{int(time.time())}.txt")
                    with open(dump_file, "w", encoding="utf-8") as fh:
                        fh.write(response_text)
                    logging.info("Wrote raw response to %s", dump_file)

                parsed = parse_numbered_list_with_years(response_text)

                for j, orig_idx in enumerate(idx_map, start=1):
                    key = str(j)
                    entry = parsed.get(key)
                    if entry:
                        school, year = entry
                    else:
                        school, year = ("N/A", "N/A")
                    # normalize
                    school = school if school and school.strip() else "N/A"
                    year = year if year and year.strip() else "N/A"
                    df.at[orig_idx, OUTPUT_COL] = school
                    df.at[orig_idx, YEAR_COL] = year
                    print(f"  {df.at[orig_idx, NAME_COL][:50]:50} -> {school} | Year: {year}")

                processed = batch_end
                save_checkpoint(processed)
                daily_requests += 1
                logging.info("Completed batch up to %d. Checkpoint saved.", processed)

                # periodic intermediate save
                try:
                    df.to_csv(OUTPUT_FILE, index=False)
                    logging.info("Intermediate save written to %s", OUTPUT_FILE)
                except Exception as e:
                    logging.warning("Failed intermediate save: %s", e)

                time.sleep(SLEEP_SECONDS + random.uniform(0, 1.5))

            else:
                # persistent failure for this batch: mark rows as N/A and continue
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
        logging.info("User interrupted run (KeyboardInterrupt). Saving progress up to %d and exiting.", processed)
        try:
            df.to_csv(OUTPUT_FILE, index=False)
        except Exception:
            pass
        print("\nInterrupted by user. Progress saved.")
        return

    # Final save
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
