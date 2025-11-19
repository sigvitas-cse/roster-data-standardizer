# education_search.py

import time
import re
import logging
from google import genai
from dotenv import load_dotenv
import os
import pandas as pd
import tenacity

# --- Setup and Initialization ---
# Set up logging
logging.basicConfig(
    filename='education_search.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()
try:
    # Initialize the Gemini Client
    client = genai.Client() 
except Exception as e:
    logging.error(f"Failed to initialize Gemini Client. Check API Key: {e}")
    # Consider raising an error or exiting if client cannot be initialized

# --- Config ---
INPUT_FILE = 'input_data.xlsx'
OUTPUT_FILE = 'output_with_education_ai.csv'
CHECKPOINT_FILE = 'education_search_checkpoint.txt'
BATCH_SIZE = 50 # Reduced batch size for name/education lookup
MODEL = 'gemini-2.5-flash'
DAILY_LIMIT = 200 # Set a cautious daily limit for API calls
SLEEP_SECONDS = 5 # Respectful sleep time
OUTPUT_COL_NAME = 'Education (AI Found)'
NAME_COL = 'Name'
ORG_COL = 'Organization/Law Firm Name'


# --- Checkpoint Functions ---
def get_last_processed():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except:
                return 0
    return 0

def save_checkpoint(row_index):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(row_index))

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are an expert researcher with access to legal and professional directories up to 2025. Your task is to find the **Law School** (Juris Doctor, JD, LLM) for each person listed below.

Rules:
- Search for the person's name and organization.
- Output ONLY the name of the law school or the degree if the school is unknown.
- If no law school is found, output "N/A".
- Output format: Numbered list corresponding to the input.

Example Input/Output:
Input:
1. John Doe @ Smith & Jones LLP
2. Jane Smith @ Google LLC

Output:
1. Harvard Law School (JD)
2. Stanford Law School (JD)

People to research:\n"""

# --- API Call with Tenacity (Error Handling & Consistency) ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    retry=tenacity.retry_if_exception_type(Exception)
)
def call_api(prompt):
    """
    Calls the Gemini API with retry logic and sets temperature=0.0 for consistency.
    """
    try:
        # Configuration object for deterministic output
        config = genai.types.GenerateContentConfig(
            # Setting temperature to 0.0 ensures the model returns the most probable,
            # consistent result, reducing random variation between runs.
            temperature=0.0 
        )
        response = client.models.generate_content(
            model=MODEL, 
            contents=prompt,
            config=config # Pass the configuration here
        )
        return response.text
    except Exception as e:
        logging.error(f"API call failed: {e}")
        # Reraise the exception to trigger the tenacity retry mechanism
        raise

# --- Load Data and Prepare DataFrame ---
logging.info(f"Starting education fetch process for {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE, engine='openpyxl')
total_rows = len(df)
logging.info(f"Loaded {total_rows} rows from {INPUT_FILE}")

if OUTPUT_COL_NAME not in df.columns:
    df[OUTPUT_COL_NAME] = 'N/A'

start_row = get_last_processed()
logging.info(f"Resuming from row {start_row + 1}")

# --- Load existing output if exists and merge with main df ---
results_to_save = [] # List to hold the updated batches
if os.path.exists(OUTPUT_FILE) and start_row > 0:
    try:
        # Load existing data to preserve prior work
        existing_df = pd.read_csv(OUTPUT_FILE)
        
        # Determine which rows were already processed (up to start_row)
        if len(existing_df) > start_row:
             # If the existing file is larger than the checkpoint, use the file size
             start_row = len(existing_df)
             logging.info(f"Checkpoint file adjusted. Resuming from row {start_row + 1} based on existing output file size.")

        # Keep the already processed part of the main dataframe in memory for final save
        results_to_save.append(df.iloc[:start_row].copy())
        
    except Exception as e:
        logging.warning(f"Could not load/merge existing output: {e}. Starting from saved checkpoint: {start_row}.")
        
# --- Main Processing Loop ---
processed = start_row
daily_requests = 0


for i in range(start_row, total_rows, BATCH_SIZE):
    if daily_requests >= DAILY_LIMIT:
        logging.warning(f"Hit daily limit of {DAILY_LIMIT} requests. Process paused.")
        print("Hit daily limit. Resume tomorrow or upgrade tier.")
        break

    end_idx = min(i + BATCH_SIZE, total_rows)
    batch_df = df.iloc[i:end_idx].copy()

    # --- Prepare Batch Prompt ---
    # Combine Name and Organization for a strong search query
    batch_entries = []
    for idx, row in batch_df.iterrows():
        name = str(row[NAME_COL]).strip()
        org = str(row.get(ORG_COL, '')).strip()
        batch_entries.append(f"{idx - i + 1}. {name} @ {org}")

    prompt = PROMPT_TEMPLATE + "\n".join(batch_entries)

    # --- Call API ---
    logging.info(f"Processing batch {i//BATCH_SIZE + 1} (rows {i+1} to {end_idx})")
    print(f"\nProcessing batch {i//BATCH_SIZE + 1} (rows {i+1} to {end_idx})")

    try:
        response_text = call_api(prompt)
        
        # --- Parse Response ---
        # Regex to find: (1). (The school name)
        parsed = dict(re.findall(r'^(\d+)\.\s*(.+)$', response_text, re.MULTILINE))
        
        # --- Update DataFrame and Results ---
        for idx, row in batch_df.iterrows():
            orig_idx = idx - i + 1 # 1-based index within the batch
            found_edu = parsed.get(str(orig_idx), 'N/A').strip()
            
            # Update the specific cell in the batch_df copy
            row[OUTPUT_COL_NAME] = found_edu
            print(f"   {row[NAME_COL][:30]}... -> {found_edu}")

        results_to_save.append(batch_df) # Append the updated batch
        
        processed = end_idx
        save_checkpoint(processed)
        
        daily_requests += 1
        logging.info(f"Completed batch {i//BATCH_SIZE + 1}. Processed {processed}/{total_rows} rows.")
        print(f"Completed batch {i//BATCH_SIZE + 1} & checkpoint saved.")

        # Save cumulative results periodically
        final_output_df = pd.concat(results_to_save, ignore_index=True)
        final_output_df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Intermediate save completed to {OUTPUT_FILE}")
        
        time.sleep(SLEEP_SECONDS)
        
    except Exception as e:
        logging.error(f"Error in batch {i//BATCH_SIZE + 1} - Skipping batch: {e}")
        # If the API call fails, treat the batch rows as is (Education='N/A' or previous value)
        results_to_save.append(batch_df) 
        time.sleep(SLEEP_SECONDS * 2) # Wait longer after a failure

# --- Final Save ---
final_output_df = pd.concat(results_to_save, ignore_index=True)
logging.info(f"Final save of results to {OUTPUT_FILE}")
final_output_df.to_csv(OUTPUT_FILE, index=False)
logging.info(f"Process completed. Total rows processed: {processed}")
print("\n--- Process Finished ---")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Total rows processed: {processed}")
print("You can resume by running the script again.")