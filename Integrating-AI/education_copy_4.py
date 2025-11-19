# education_search.py

import time
import re
import logging
import os
import pandas as pd
import tenacity
import json
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List

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
    raise # Exit if client cannot be initialized

# --- Config ---
INPUT_FILE = 'input_data.xlsx'
OUTPUT_FILE = 'output_with_education_ai.csv'
CHECKPOINT_FILE = 'education_search_checkpoint.txt'
BATCH_SIZE = 50 
MODEL = 'gemini-2.5-flash'
DAILY_LIMIT = 200 
SLEEP_SECONDS = 5 
OUTPUT_COL_NAME = 'Education (AI Found)'
NAME_COL = 'Name'
ORG_COL = 'Organization/Law Firm Name'


# --- Pydantic Schema for Structured Output ---
# Define the structure the model must return
class PersonEducation(BaseModel):
    """Schema for a single person's education result."""
    # Use the 'Field' description to give the model context for each field
    id: int = Field(description="The 1-based index corresponding to the input list (e.g., 1, 2, 3...).")
    education: str = Field(description="The Law School and degree (e.g., 'Harvard Law School (JD)'). If the information is not highly confident for this person, the value MUST be 'N/A'.")

class BatchResponse(BaseModel):
    """Schema for the entire batch response."""
    results: List[PersonEducation] = Field(description="A list of the education results for all people in the batch, matching the input order.")


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

# --- Prompt Template (Optimized for JSON Output) ---
PROMPT_TEMPLATE = """You are an expert researcher with access to legal and professional directories up to 2025. Your task is to find the **Law School** (Juris Doctor, JD, LLM, LL.B.) for each person listed below.

Rules:
1. Search for the person's name and organization.
2. The response MUST be a JSON object that strictly adheres to the provided schema.
3. For the 'education' field, output ONLY the name of the law school and the degree (e.g., "Pepperdine University School of Law (JD)").
4. **CRITICAL RULE:** If you are not highly confident in the match for the person's education, or if no law school is found, you **MUST** output **"N/A"** for the 'education' field.

People to research:\n"""

# --- API Call with Tenacity (Error Handling & Structured Consistency) ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    retry=tenacity.retry_if_exception_type(Exception)
)
def call_api(prompt):
    """
    Calls the Gemini API with retry logic, sets temperature=0.0 for consistency,
    and enforces JSON output via Pydantic schema.
    """
    try:
        # Configuration object for deterministic and structured output
        config = genai.types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            # Pass the Pydantic schema to enforce the JSON structure
            response_schema=BatchResponse,
        )
        response = client.models.generate_content(
            model=MODEL, 
            contents=prompt,
            config=config
        )
        # Validate the raw JSON text against the schema immediately
        return BatchResponse.model_validate_json(response.text)
    except Exception as e:
        logging.error(f"API call failed or JSON validation failed: {e}")
        raise # Reraise to trigger tenacity retry

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
results_to_save = [] 
if os.path.exists(OUTPUT_FILE) and start_row > 0:
    try:
        existing_df = pd.read_csv(OUTPUT_FILE)
        
        if len(existing_df) > start_row:
             start_row = len(existing_df)
             logging.info(f"Checkpoint file adjusted. Resuming from row {start_row + 1} based on existing output file size.")

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
    batch_entries = []
    for idx, row in batch_df.iterrows():
        name = str(row[NAME_COL]).strip()
        org = str(row.get(ORG_COL, '')).strip()
        # The index (1, 2, 3...) is CRITICAL for mapping the JSON output back
        batch_entries.append(f"{idx - i + 1}. {name} @ {org}")

    prompt = PROMPT_TEMPLATE + "\n".join(batch_entries)

    # --- Call API ---
    logging.info(f"Processing batch {i//BATCH_SIZE + 1} (rows {i+1} to {end_idx})")
    print(f"\nProcessing batch {i//BATCH_SIZE + 1} (rows {i+1} to {end_idx})")

    try:
        # call_api now returns a Pydantic object (BatchResponse)
        response_data: BatchResponse = call_api(prompt)
        
        # --- Update DataFrame and Results from Structured Output ---
        # Convert the list of Pydantic PersonEducation models to a dictionary for easy lookup
        parsed_results = {item.id: item.education for item in response_data.results}
        
        for idx, row in batch_df.iterrows():
            orig_idx = idx - i + 1 # 1-based index within the batch
            
            # Use the ID to get the education result
            found_edu = parsed_results.get(orig_idx, 'N/A').strip()
            
            # Update the specific cell in the batch_df copy
            row[OUTPUT_COL_NAME] = found_edu
            print(f"   {row[NAME_COL][:30]}... -> {found_edu}")

        results_to_save.append(batch_df) 
        
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
        results_to_save.append(batch_df) 
        time.sleep(SLEEP_SECONDS * 2) 

# --- Final Save ---
final_output_df = pd.concat(results_to_save, ignore_index=True)
logging.info(f"Final save of results to {OUTPUT_FILE}")
final_output_df.to_csv(OUTPUT_FILE, index=False)
logging.info(f"Process completed. Total rows processed: {processed}")
print("\n--- Process Finished ---")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Total rows processed: {processed}")
print("You can resume by running the script again.")