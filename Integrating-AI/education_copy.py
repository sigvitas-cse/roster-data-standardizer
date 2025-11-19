# education_search_fixed.py
"""
Robust batch education lookup using the Google/genai client.

Notes:
- Expects an input Excel file with columns NAME_COL and ORG_COL (defaults below).
- Writes output CSV and a checkpoint file to resume progress.
- Uses tenacity for retrying API calls.
- Adjust MODEL, BATCH_SIZE, and DAILY_LIMIT as needed.
"""

import os
import time
import re
import logging
import sys
from dotenv import load_dotenv
import pandas as pd
import tenacity

# Optional: import genai (Google Gemini client). If not installed, script logs error.
try:
    from google import genai
except Exception:
    genai = None

# ----------------- Configuration -----------------
INPUT_FILE = "input_data.xlsx"
OUTPUT_FILE = "output_with_education_ai.csv"
CHECKPOINT_FILE = "education_search_checkpoint.txt"
LOG_FILE = "education_search.log"

BATCH_SIZE = 50
MODEL = "gemini-2.5-flash"  # change if your account uses a different model
DAILY_LIMIT = 200
SLEEP_SECONDS = 5

NAME_COL = "Name"
ORG_COL = "Organization/Law Firm Name"
OUTPUT_COL = "Education (AI Found)"

# Toggle to True to dump the raw model response to a debug file for troubleshooting
DUMP_RAW_RESPONSE = False
RAW_RESPONSE_DUMP_DIR = "raw_responses"

# ----------------- Logging Setup -----------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# ----------------- Load environment and initialize client -----------------
load_dotenv()

client = None
if genai is not None:
    try:
        client = genai.Client()  # ensure GOOGLE_API_KEY (or appropriate key) is set in env
        logging.info("Initialized genai client.")
    except Exception as e:
        logging.error(f"Failed to initialize genai client: {e}")
        client = None
else:
    logging.warning("google.genai package not available. `client` will be None. Install via docs.")

# ----------------- Prompt Template -----------------
PROMPT_TEMPLATE = """You are an expert researcher with access to legal and professional directories up to 2025.
Task: For each person below, find the Law School (Juris Doctor, JD, LLM). Output ONLY the law school name and degree (e.g., "Pepperdine University (JD)").
If no law school found, output "N/A".

Output format:
1. <Law school or N/A>
2. <Law school or N/A>

People to research:
"""

# ----------------- Utilities: Checkpoint -----------------
def get_last_processed():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                content = f.read().strip()
                return int(content) if content else 0
        except Exception as e:
            logging.warning(f"Could not read checkpoint file: {e}")
            return 0
    return 0

def save_checkpoint(row_index):
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(row_index))
    except Exception as e:
        logging.warning(f"Failed to save checkpoint: {e}")

# ----------------- Utilities: Response extraction & parsing -----------------
def safe_get_response_text(response):
    """Extract a string from various response shapes."""
    if response is None:
        return ""
    # Some SDKs return a simple object with `text`
    try:
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text
    except Exception:
        pass
    # Other SDKs may return choices, content, or nested structures
    for attr in ("content", "output", "answer", "response", "result"):
        try:
            if hasattr(response, attr):
                val = getattr(response, attr)
                if isinstance(val, str):
                    return val
                # sometimes it's a list or dict
                if isinstance(val, (list, tuple)):
                    # join textual parts
                    return "\n".join(map(str, val))
                return str(val)
        except Exception:
            continue
    # fallback: string representation
    return str(response)

def parse_numbered_list(text):
    """
    Parse lines like:
      1. Pepperdine University (JD)
      2) Stanford Law School (JD)
      3 - N/A
    Returns a dict mapping '1'->'Pepperdine University (JD)', etc.
    """
    parsed = {}
    if not text:
        return parsed

    # Optionally trim preamble: find first numbered line
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    # find index of first numbered pattern
    start_idx = 0
    for i, ln in enumerate(lines):
        if re.match(r'^\s*\d+\s*[\.\)\-:]\s*', ln) or re.match(r'^\s*\d+\s+', ln):
            start_idx = i
            break
    lines = lines[start_idx:]

    pattern = re.compile(r'^\s*(\d+)\s*[\.\)\-:]\s*(.+)$')  # matches "1. text", "1) text", "1 - text"
    fallback_pattern = re.compile(r'^\s*(\d+)\s+(.+)$')     # matches "1 text"

    for ln in lines:
        m = pattern.match(ln)
        if m:
            parsed[m.group(1)] = m.group(2).strip()
            continue
        m2 = fallback_pattern.match(ln)
        if m2:
            parsed[m2.group(1)] = m2.group(2).strip()
            continue
        # If a line doesn't match, try to find a leading number anywhere
        m3 = re.match(r'^\s*(\d+)', ln)
        if m3:
            key = m3.group(1)
            # use remainder of the line as value if present
            remainder = ln[m3.end():].strip(" .:-)\t")
            parsed[key] = remainder if remainder else ""
    return parsed

# ----------------- API Call w/ Tenacity -----------------
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    retry=tenacity.retry_if_exception_type(Exception),
)
def call_api(prompt: str, model: str = MODEL, temperature: float = 0.0):
    """Call the genai client with a deterministic config; robust to multiple response shapes."""
    if client is None:
        raise RuntimeError("genai client not initialized. Install and configure credentials.")
    try:
        # Example for google.genai client; different client versions may differ.
        config = genai.types.GenerateContentConfig(temperature=temperature)
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        return response
    except Exception as e:
        logging.error(f"API call failed: {e}")
        raise

# ----------------- Main script -----------------
def main():
    logging.info(f"Starting education fetch process for {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file not found: {INPUT_FILE}")
        return

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    total_rows = len(df)
    logging.info(f"Loaded {total_rows} rows from {INPUT_FILE}")

    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = "N/A"

    # load checkpoint and existing output if present
    start_row = get_last_processed()
    logging.info(f"Resuming from row (0-based) {start_row}")

    # If there's an existing output file, load and align to avoid reprocessing
    if os.path.exists(OUTPUT_FILE) and start_row == 0:
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            # if existing has same length assume it's the previous run; use it to resume
            if len(existing) <= len(df):
                # try to copy existing column to df for continuity
                if OUTPUT_COL in existing.columns:
                    df.loc[: len(existing) - 1, OUTPUT_COL] = existing.loc[: len(existing) - 1, OUTPUT_COL].values
                    start_row = len(existing)
                    logging.info(f"Adjusted start_row to {start_row} based on existing output file.")
        except Exception as e:
            logging.warning(f"Could not load existing output file: {e}")

    processed = start_row
    daily_requests = 0

    # Ensure directory for raw response dumps
    if DUMP_RAW_RESPONSE and not os.path.isdir(RAW_RESPONSE_DUMP_DIR):
        os.makedirs(RAW_RESPONSE_DUMP_DIR, exist_ok=True)

    for batch_start in range(start_row, total_rows, BATCH_SIZE):
        if daily_requests >= DAILY_LIMIT:
            logging.warning("Reached DAILY_LIMIT of requests, stopping.")
            print("Hit daily limit. Resume later.")
            break

        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_df = df.iloc[batch_start:batch_end].copy()

        # Build prompt lines with 1..N within batch, keep mapping to original indices
        batch_lines = []
        idx_map = []
        for j, (orig_index, row) in enumerate(batch_df.iterrows(), start=1):
            name = str(row.get(NAME_COL, "")).strip()
            org = str(row.get(ORG_COL, "")).strip()
            batch_lines.append(f"{j}. {name} @ {org}")
            idx_map.append(orig_index)

        prompt = PROMPT_TEMPLATE + "\n".join(batch_lines)

        logging.info(f"Processing batch rows {batch_start}..{batch_end-1} (total {len(idx_map)})")
        print(f"\nProcessing rows {batch_start} to {batch_end-1}")

        try:
            response = call_api(prompt)
            response_text = safe_get_response_text(response)

            # Optional dump for debugging raw response
            if DUMP_RAW_RESPONSE:
                dump_file = os.path.join(RAW_RESPONSE_DUMP_DIR, f"response_{batch_start}_{int(time.time())}.txt")
                with open(dump_file, "w", encoding="utf-8") as f:
                    f.write(response_text)
                logging.info(f"Wrote raw response to {dump_file}")

            parsed = parse_numbered_list(response_text)

            # Update the main DataFrame using original indices
            for j, orig_idx in enumerate(idx_map, start=1):
                key = str(j)
                found = parsed.get(key, "N/A").strip() if parsed else "N/A"
                # final normalization: if empty string set to N/A
                if found == "" or found.lower() == "none":
                    found = "N/A"
                df.at[orig_idx, OUTPUT_COL] = found
                print(f"  {df.at[orig_idx, NAME_COL][:50]:50} -> {found}")

            processed = batch_end
            save_checkpoint(processed)
            daily_requests += 1
            logging.info(f"Completed batch up to row {processed}. Checkpoint saved.")

            # Periodic save: write the whole DataFrame to output so far
            try:
                df.to_csv(OUTPUT_FILE, index=False)
                logging.info(f"Intermediate save written to {OUTPUT_FILE}")
            except Exception as e:
                logging.warning(f"Unable to write intermediate output to {OUTPUT_FILE}: {e}")

            time.sleep(SLEEP_SECONDS)

        except Exception as exc:
            logging.error(f"Batch starting at {batch_start} failed: {exc}", exc_info=True)
            # mark rows in this batch as N/A if they are blank
            for orig_idx in idx_map:
                if not df.at[orig_idx, OUTPUT_COL] or pd.isna(df.at[orig_idx, OUTPUT_COL]):
                    df.at[orig_idx, OUTPUT_COL] = "N/A"
            # save progress so far
            try:
                df.to_csv(OUTPUT_FILE, index=False)
            except Exception:
                pass
            # wait longer on error
            time.sleep(SLEEP_SECONDS * 2)

    # Final save
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Final output written to {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Failed to write final output: {e}")

    logging.info(f"Job finished. Processed up to row {processed} of {total_rows}.")
    print("\n--- Process Finished ---")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"Total rows processed: {processed}")

if __name__ == "__main__":
    main()
