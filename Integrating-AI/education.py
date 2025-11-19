# education_search_fixed_v4.py
"""
Robust batch education lookup using the Google/genai client.
Version 4:
- Configured for 'gemini-2.0-flash' based on user's available model list.
- Includes Rate Limit (429) handling (Auto-sleep).
- Includes Auto-save and Resume.
"""

import os
import time
import re
import logging
import sys
from dotenv import load_dotenv
import pandas as pd

# ----------------- Imports -----------------
try:
    from google import genai
    from google.genai.errors import ClientError
except ImportError:
    print("CRITICAL: google.genai package not found.")
    print("Please run: pip install google-genai")
    sys.exit(1)

# ----------------- Configuration -----------------
INPUT_FILE = "input_data.xlsx"
OUTPUT_FILE = "output_with_education_ai.csv"
CHECKPOINT_FILE = "education_search_checkpoint.txt"
LOG_FILE = "education_search.log"

# UPDATED: Based on your diagnostic logs, this model is available.
# If this hits a limit too quickly, change this to "gemini-2.0-flash-lite"
MODEL = "gemini-2.0-flash"

# Batch size optimization
BATCH_SIZE = 75
SLEEP_SECONDS = 2   # Pause between batches

NAME_COL = "Name"
ORG_COL = "Organization/Law Firm Name"
OUTPUT_COL = "Education (AI Found)"

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

# ----------------- Initialization -----------------
load_dotenv()

try:
    client = genai.Client()
    logging.info("Initialized genai client.")
except Exception as e:
    logging.error(f"Failed to initialize client: {e}")
    sys.exit(1)

# ----------------- Prompt Template -----------------
PROMPT_TEMPLATE = """You are an expert researcher.
Task: For each person below, find the Law School (Juris Doctor, JD, LLM). 
Output ONLY the law school name and degree (e.g., "Pepperdine University (JD)").
If no law school found, output "N/A".

Output format:
1. <Law school or N/A>
2. <Law school or N/A>

People to research:
"""

# ----------------- Utilities -----------------
def get_last_processed():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                content = f.read().strip()
                return int(content) if content else 0
        except Exception:
            return 0
    return 0

def save_checkpoint(row_index):
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(row_index))
    except Exception as e:
        logging.warning(f"Failed to save checkpoint: {e}")

def safe_get_response_text(response):
    if not response: return ""
    try:
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text
    except: pass
    return str(response)

def parse_numbered_list(text):
    parsed = {}
    if not text: return parsed
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pattern = re.compile(r'^(\d+)[\.\)\-:]\s*(.+)$')
    fallback = re.compile(r'^(\d+)\s+(.+)$')

    for ln in lines:
        m = pattern.match(ln) or fallback.match(ln)
        if m:
            parsed[m.group(1)] = m.group(2).strip()
    return parsed

# ----------------- Diagnostic Tool -----------------
def verify_model_access():
    """
    Tries to ping the model before starting the heavy job.
    """
    print(f"\n--- Diagnosing Model Access: {MODEL} ---")
    try:
        # Try a cheap 1-token generation
        client.models.generate_content(model=MODEL, contents="Test")
        print("✅ Model access confirmed. Starting job...\n")
        return True
    except ClientError as e:
        print(f"❌ ERROR: Model '{MODEL}' failed.")
        print(f"Details: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error during check: {e}")
        return False

# ----------------- Robust API Call -----------------
def call_api_robust(prompt, model=MODEL):
    """
    Handles 429 Rate Limits by sleeping instead of crashing.
    """
    max_retries = 5
    base_wait = 30
    
    for attempt in range(max_retries):
        try:
            config = genai.types.GenerateContentConfig(temperature=0.0)
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            return response
            
        except ClientError as e:
            # 429 = Resource Exhausted (Rate Limit)
            if e.code == 429 or "429" in str(e):
                wait_time = base_wait * (attempt + 1)
                logging.warning(f"Rate Limit Hit (429). Sleeping {wait_time}s...")
                print(f"   [!] Rate limit hit. Pausing for {wait_time}s...")
                time.sleep(wait_time)
            elif e.code == 404:
                logging.error("Model 404'd during execution.")
                raise e
            else:
                logging.error(f"API Error: {e}")
                time.sleep(5)
                if attempt == max_retries - 1: raise e
                
        except Exception as e:
            logging.error(f"Network Error: {e}")
            time.sleep(10)
            if attempt == max_retries - 1: raise e
            
    raise Exception("Max retries exceeded. Daily quota likely exhausted.")

# ----------------- Main -----------------
def main():
    # 1. Verify Model before doing anything
    if not verify_model_access():
        print("Exiting due to model configuration error.")
        sys.exit(1)

    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    total_rows = len(df)
    
    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = "N/A"

    # 2. Resume Logic
    start_row = get_last_processed()
    if os.path.exists(OUTPUT_FILE) and start_row > 0:
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            common = min(len(df), len(existing))
            df.loc[:common-1, OUTPUT_COL] = existing.loc[:common-1, OUTPUT_COL].values
            logging.info(f"Merged existing output. Resuming at row {start_row}")
        except Exception:
            logging.warning("Could not merge existing CSV. Proceeding.")

    processed = start_row
    print(f"Processing {total_rows} rows (Batch Size: {BATCH_SIZE})...")

    # 3. Processing Loop
    for batch_start in range(start_row, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_df = df.iloc[batch_start:batch_end]
        
        batch_lines = []
        idx_map = []
        
        for j, (orig_index, row) in enumerate(batch_df.iterrows(), start=1):
            name = str(row.get(NAME_COL, "")).strip()
            org = str(row.get(ORG_COL, "")).strip()
            batch_lines.append(f"{j}. {name} @ {org}")
            idx_map.append(orig_index)

        prompt = PROMPT_TEMPLATE + "\n".join(batch_lines)
        print(f"Batch {batch_start}-{batch_end-1}...", end=" ", flush=True)
        
        try:
            response = call_api_robust(prompt)
            response_text = safe_get_response_text(response)
            parsed = parse_numbered_list(response_text)

            for j, orig_idx in enumerate(idx_map, start=1):
                key = str(j)
                found = parsed.get(key, "N/A").strip()
                if found.lower() in ["", "none", "n/a"]: found = "N/A"
                df.at[orig_idx, OUTPUT_COL] = found

            print("Done.")
            processed = batch_end
            save_checkpoint(processed)
            df.to_csv(OUTPUT_FILE, index=False)
            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print("\nUser stopped script.")
            break
        except Exception as e:
            logging.error(f"Batch failed: {e}")
            print(f"\nError: {e}")
            print("Saving progress and exiting.")
            df.to_csv(OUTPUT_FILE, index=False)
            break

    df.to_csv(OUTPUT_FILE, index=False)
    logging.info("Job Complete.")
    print(f"\nFinished. Processed {processed}/{total_rows}.")

if __name__ == "__main__":
    main()