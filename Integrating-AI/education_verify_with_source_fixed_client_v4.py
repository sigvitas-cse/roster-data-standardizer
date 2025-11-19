#!/usr/bin/env python3
"""
education_verify_with_source_fixed_client_v5.py

This is a full, self-contained verifier script that:
- Requires GENAI_API_KEY or GOOGLE_API_KEY in environment (explicitly)
- Uses Google GenAI to extract education + source and verifies by fetching pages
- Includes robust retries/backoff and checkpointing
- Produces output CSV and audit CSV
- Use --only-audit to generate audit from existing output without calling the API
"""

import os
import sys
import time
import re
import random
import argparse
import logging
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# try import genai
try:
    from google import genai
except Exception:
    genai = None

# ---------------- Config ----------------
INPUT_FILE = "input_data.xlsx"
OUTPUT_FILE = "output_with_education_verified_v5.csv"
AUDIT_FILE = "audit_unverified_v5.csv"
CHECKPOINT_FILE = "education_checkpoint_v5.txt"
RAW_RESP_DIR = "raw_responses_verified_v5"
RAW_PAGES_DIR = "raw_pages_v5"
LOG_FILE = "education_verify_v5.log"

MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.0

DEFAULT_BATCH_SIZE = 1
MAX_ATTEMPTS = 6
BACKOFF_BASE = 4
BACKOFF_MAX = 180
SLEEP_SECONDS = 1.2

NAME_COL = "Name"
ORG_COL = "Organization/Law Firm Name"

COL_EDU = "Education (AI Found)"
COL_SRC = "Education Source (AI Found)"
COL_VER = "Source Verified (Yes/No/N/A/Trusted)"
COL_YEAR = "Education Year (AI Found)"
COL_CONF = "AI_Confidence"

HTTP_TIMEOUT = 12
HTTP_HEADERS = {"User-Agent": "education-verifier/1.0 (+https://example.com)"}

NO_FETCH_DOMAINS = {"linkedin.com", "www.linkedin.com", "linkedin.cn"}

FETCH_AND_VERIFY_DOMAINS = {
    "martindale.com", "www.martindale.com",
    "justia.com", "www.justia.com",
    "kirkland.com", "www.kirkland.com",
    "mofo.com", "www.mofo.com",
    "mbhb.com", "www.mbhb.com",
    "knobbe.com", "www.knobbe.com",
    "notarolaw.com", "www.notarolaw.com",
    "pillsburylaw.com", "www.pillsburylaw.com",
}

TRUSTED_ASSUME_DOMAINS = {"gov.example"}  # add your trusted domains here if you want

FUZZY_THRESHOLD = 0.22

# ---------------- Logging ----------------
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)

# ---------------- Utilities ----------------
def ensure_dirs():
    os.makedirs(RAW_RESP_DIR, exist_ok=True)
    os.makedirs(RAW_PAGES_DIR, exist_ok=True)

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

def domain_of(url):
    if not url or str(url).strip().upper() == "N/A":
        return None
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc
    except Exception:
        return None

def safe_get_response_text(response):
    if response is None:
        return ""
    try:
        if hasattr(response, "text"):
            return response.text
    except Exception:
        pass
    return str(response)

def parse_numbered_list(text):
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
    if current_key is not None:
        parsed[current_key] = " ".join([p for p in current_parts if p is not None]).strip()
    return {k: v.strip() for k, v in parsed.items()}

def extract_edu_and_source(raw_line):
    if not raw_line or raw_line.strip().upper() == "N/A":
        return "N/A", "N/A"
    m = re.search(r'\(source\s*:\s*(.*?)\)\s*$', raw_line, flags=re.IGNORECASE)
    if m:
        src = m.group(1).strip()
        edu = re.sub(r'\(source\s*:\s*.*?\)\s*$', '', raw_line, flags=re.IGNORECASE).strip()
        return edu if edu else "N/A", src if src else "N/A"
    return raw_line.strip(), "N/A"

def extract_jd_year(full_text):
    if not full_text or full_text.strip().upper() == "N/A":
        return "N/A"
    parts = [p.strip() for p in re.split(r';\s*', full_text) if p.strip()]
    for p in parts:
        if re.search(r'\bJ\.?\s*D\.?\b', p, flags=re.IGNORECASE):
            m = re.search(r'\b(19|20)\d{2}\b', p)
            if m:
                return m.group(0)
    m2 = re.search(r'\b(19|20)\d{2}\b', full_text)
    if m2:
        return m2.group(0)
    return "N/A"

def save_raw_response(text, batch_start):
    ensure_dirs()
    ts = int(time.time())
    path = os.path.join(RAW_RESP_DIR, f"resp_{batch_start}_{ts}.txt")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        logging.info("Saved raw response to %s", path)
    except Exception as e:
        logging.warning("Failed to save raw response: %s", e)

def save_fetched_page(html, domain):
    ensure_dirs()
    try:
        fname = re.sub(r'[^0-9a-zA-Z_\-\.]', '_', domain) + "_" + str(int(time.time())) + ".html"
        path = os.path.join(RAW_PAGES_DIR, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logging.info("Saved fetched page to %s", path)
        return path
    except Exception as e:
        logging.debug("Failed to save fetched page: %s", e)
        return None

def fuzzy_similarity(a, b):
    import difflib
    if not a or not b:
        return 0.0
    try:
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0

def fetch_and_verify_page(url, expected_text, assume_trusted=False):
    if not url or str(url).strip().upper() == "N/A":
        return "N/A"

    dom = domain_of(url)
    if dom is None:
        return False

    if any(nd in dom for nd in NO_FETCH_DOMAINS):
        logging.info("Domain %s in NO_FETCH_DOMAINS; skipping fetch", dom)
        return "NO_FETCH"

    if assume_trusted and dom in TRUSTED_ASSUME_DOMAINS:
        logging.info("Domain %s in TRUSTED_ASSUME_DOMAINS and assume_trusted on; marking Trusted", dom)
        return True

    try:
        parsed = urlparse(url)
        url_try = url if parsed.scheme else "https://" + url.lstrip("/")
        r = requests.get(url_try, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200 and url_try.startswith("https://"):
            try:
                url_try2 = "http://" + url_try[len("https://"):]
                r2 = requests.get(url_try2, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
                if r2.status_code == 200:
                    page_html = r2.text
                else:
                    logging.debug("Fetch failed both https and http for %s: %s / %s", url, r.status_code, r2.status_code)
                    return False
            except Exception as e:
                logging.debug("HTTP fallback fetch failed: %s", e)
                return False
        else:
            page_html = r.text
    except Exception as e:
        logging.debug("Fetch exception for %s: %s", url, e)
        return False

    save_fetched_page(page_html, dom)

    if not expected_text or expected_text.strip().upper() == "N/A":
        return "N/A"

    try:
        soup = BeautifulSoup(page_html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        page_text = soup.get_text(separator=" ", strip=True).lower()
    except Exception:
        page_text = page_html.lower()

    raw = re.sub(r'[\(\)\[\]\|]', ' ', expected_text)
    cand = re.split(r'[;,\-–—,]', raw)
    tokens = [c.strip().lower() for c in cand if len(c.strip()) >= 4]

    if tokens:
        for tk in tokens:
            words = [w for w in re.findall(r'\w+', tk) if len(w) > 2]
            if not words:
                continue
            if all(w in page_text for w in words):
                return True

    sample = page_text[:8000]
    sim = fuzzy_similarity(expected_text.lower(), sample)
    logging.debug("Fuzzy sim for %s => %.3f", url, sim)
    return sim >= FUZZY_THRESHOLD

# ---------------- genai client init & safe call (STRICT — requires API key) ----------------
def init_genai_client():
    """
    Strict initialization: we require GENAI_API_KEY or GOOGLE_API_KEY to be present.
    This avoids the vague `Client()` default-init failure you saw.
    """
    if genai is None:
        return None, "google-genai SDK is not installed. Run: pip install google-genai"
    api_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        msg = ("Missing GENAI_API_KEY / GOOGLE_API_KEY environment variable.\n"
               "Set your API key and re-run. Examples:\n\n"
               "Windows PowerShell (temporary):\n  $env:GENAI_API_KEY = \"sk-xxx\"\n  py education_verify_with_source_fixed_client_v5.py\n\n"
               "Windows cmd (temporary):\n  set GENAI_API_KEY=sk-xxx\n  py education_verify_with_source_fixed_client_v5.py\n\n"
               "Windows (persist):\n  setx GENAI_API_KEY \"sk-xxx\"\n  (then restart terminal)\n\n"
               "Linux/macOS:\n  export GENAI_API_KEY=\"sk-xxx\"\n  py education_verify_with_source_fixed_client_v5.py\n\n"
               "If you prefer Google Cloud/Vertex authentication, configure the Vertex credentials per google-genai docs.\n")
        return None, msg
    try:
        client = genai.Client(api_key=api_key)
        logging.info("Initialized genai client using GENAI_API_KEY environment variable.")
        return client, None
    except Exception as e:
        logging.error("genai.Client init failed with provided api_key: %s", e, exc_info=True)
        return None, str(e)

def call_model_with_retries(client, prompt, max_attempts=MAX_ATTEMPTS):
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            try:
                config = genai.types.GenerateContentConfig(temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS)
                resp = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            except Exception:
                resp = client.models.generate_content(model=MODEL, contents=prompt)
            text = safe_get_response_text(resp)
            return True, text
        except Exception as exc:
            msg = str(exc)
            logging.warning("Model attempt %d/%d failed: %s", attempt, max_attempts, msg.splitlines()[0] if msg else msg)
            if any(x in msg.lower() for x in ("503", "service unavailable", "429", "rate limit", "quota", "unavailable")):
                if attempt >= max_attempts:
                    return False, msg
                backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))) + random.uniform(0, 2.0)
                logging.info("Transient error: backing off %.1f s", backoff)
                time.sleep(backoff)
                continue
            else:
                logging.error("Non-transient model error: %s", msg)
                return False, msg
    return False, "Unknown failure"

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Education verify with source (v5)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for model queries")
    parser.add_argument("--assume-trusted", action="store_true", help="Assume TRUSTED_ASSUME_DOMAINS are Trusted without fetch")
    parser.add_argument("--start-row", type=int, default=None, help="Override checkpoint start row (0-based)")
    parser.add_argument("--only-audit", action="store_true", help="Do not call model; only produce audit CSV from existing output")
    args = parser.parse_args()

    ensure_dirs()
    client, err = init_genai_client()
    if client is None and not args.only_audit:
        logging.error("genai client not initialized: %s", err)
        print(err)
        return

    if not os.path.exists(INPUT_FILE):
        logging.error("Input file not found: %s", INPUT_FILE)
        print("Input file not found:", INPUT_FILE)
        return

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    total = len(df)
    logging.info("Loaded %d rows from %s", total, INPUT_FILE)

    for c in (COL_EDU, COL_SRC, COL_VER, COL_YEAR, COL_CONF):
        if c not in df.columns:
            df[c] = "N/A"

    start = get_last_processed()
    if args.start_row is not None:
        start = args.start_row
    logging.info("Resuming from row (0-based): %d", start)
    processed = start

    if args.only_audit:
        if os.path.exists(OUTPUT_FILE):
            out_df = pd.read_csv(OUTPUT_FILE)
            mask = ~out_df[COL_VER].astype(str).str.lower().eq("yes")
            audit_df = out_df.loc[mask, [NAME_COL, ORG_COL, COL_EDU, COL_SRC, COL_VER, COL_CONF, COL_YEAR]]
            audit_df.to_csv(AUDIT_FILE, index=False)
            logging.info("Audit file written to %s (rows where Verified != Yes)", AUDIT_FILE)
            print("Audit file created:", AUDIT_FILE)
        else:
            print("No output file found to audit:", OUTPUT_FILE)
        return

    try:
        batch_size = max(1, args.batch_size)
        for batch_start in range(start, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_df = df.iloc[batch_start:batch_end].copy()
            lines = []
            index_map = []
            for j, (orig_idx, row) in enumerate(batch_df.iterrows(), start=1):
                name = str(row.get(NAME_COL, "")).strip()
                org = str(row.get(ORG_COL, "")).strip() if ORG_COL in df.columns else ""
                lines.append(f"{j}. {name} @ {org}")
                index_map.append(orig_idx)

            prompt = (
                "You are an expert researcher with access to public directories up to 2025.\n"
                "For each person below, list all verified education entries and include a short source URL.\n"
                "Format EXACTLY (one line per numbered item):\n"
                "1. <entries separated by ';'> (source: <short url or N/A>)\n"
                "If nothing credible is found: N/A (source: N/A)\nPeople to research:\n"
            )
            prompt += "\n".join(lines)

            logging.info("Querying rows %d..%d", batch_start, batch_end - 1)
            ok, resp_text = call_model_with_retries(client, prompt)
            if not ok:
                logging.error("Persistent model failure for batch %d..%d : %s", batch_start, batch_end - 1, resp_text)
                for orig_idx in index_map:
                    df.at[orig_idx, COL_EDU] = "N/A"
                    df.at[orig_idx, COL_SRC] = "N/A"
                    df.at[orig_idx, COL_VER] = "N/A"
                    df.at[orig_idx, COL_YEAR] = "N/A"
                    df.at[orig_idx, COL_CONF] = "low"
                processed = batch_end
                save_checkpoint(processed)
                df.to_csv(OUTPUT_FILE, index=False)
                backoff = min(BACKOFF_MAX, 30 + random.uniform(0, 30))
                logging.info("Backing off %.1f s after persistent model failure", backoff)
                time.sleep(backoff)
                continue

            save_raw_response(resp_text, batch_start)
            parsed = parse_numbered_list(resp_text)

            for j, orig_idx in enumerate(index_map, start=1):
                key = str(j)
                raw_val = parsed.get(key, "N/A")
                edu_text, src = extract_edu_and_source(raw_val)
                year = extract_jd_year(edu_text)
                dom = domain_of(src)

                verified = "N/A"
                confidence = "low"

                if not src or src.strip().upper() == "N/A":
                    verified = "N/A"
                    confidence = "low" if edu_text.strip().upper() == "N/A" else "medium"
                else:
                    if dom and any(nd in dom for nd in NO_FETCH_DOMAINS):
                        verified = "N/A"
                        confidence = "medium"
                        logging.info("Domain %s is NO-FETCH; skipping fetch", dom)
                    elif dom and dom in FETCH_AND_VERIFY_DOMAINS:
                        v = fetch_and_verify_page(src, edu_text, assume_trusted=args.assume_trusted)
                        if v is True:
                            verified = "Yes"
                            confidence = "high"
                        elif v == "NO_FETCH":
                            verified = "N/A"
                            confidence = "medium"
                        elif v == "N/A":
                            verified = "N/A"
                            confidence = "medium"
                        else:
                            verified = "No"
                            confidence = "medium"
                    elif dom and (args.assume_trusted and dom in TRUSTED_ASSUME_DOMAINS):
                        verified = "Trusted"
                        confidence = "high"
                    else:
                        v = fetch_and_verify_page(src, edu_text, assume_trusted=args.assume_trusted)
                        if v is True:
                            verified = "Yes"
                            confidence = "high"
                        elif v == "NO_FETCH":
                            verified = "N/A"
                            confidence = "medium"
                        elif v == "N/A":
                            verified = "N/A"
                            confidence = "medium"
                        else:
                            verified = "No"
                            confidence = "medium"

                df.at[orig_idx, COL_EDU] = edu_text
                df.at[orig_idx, COL_SRC] = src
                df.at[orig_idx, COL_VER] = verified
                df.at[orig_idx, COL_YEAR] = year
                df.at[orig_idx, COL_CONF] = confidence

                preview = (edu_text[:140] + "...") if len(edu_text) > 140 else edu_text
                print(f"{batch_start + j - 1:6d}: {df.at[orig_idx, NAME_COL]:40.40s} -> {preview:60s} | src: {src:30s} | ver: {verified} | conf: {confidence}")

            processed = batch_end
            save_checkpoint(processed)
            try:
                df.to_csv(OUTPUT_FILE, index=False)
                logging.info("Intermediate save to %s", OUTPUT_FILE)
            except Exception as e:
                logging.warning("Failed intermediate save: %s", e)

            time.sleep(SLEEP_SECONDS + random.uniform(0, 1.0))

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Saving progress up to %d and exiting.", processed)
        try:
            df.to_csv(OUTPUT_FILE, index=False)
        except Exception:
            pass
        print("\nInterrupted by user. Progress saved.")
    finally:
        try:
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                    logging.info("Closed genai client.")
                except Exception:
                    pass
        except Exception:
            pass

    try:
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info("Final output saved to %s", OUTPUT_FILE)
        mask = ~df[COL_VER].astype(str).str.lower().eq("yes")
        audit_df = df.loc[mask, [NAME_COL, ORG_COL, COL_EDU, COL_SRC, COL_VER, COL_CONF, COL_YEAR]]
        audit_df.to_csv(AUDIT_FILE, index=False)
        logging.info("Audit file written to %s (rows where Verified != Yes)", AUDIT_FILE)
        print("\nDone. Output saved to", OUTPUT_FILE)
        print("Audit (unverified) saved to", AUDIT_FILE)
    except Exception as e:
        logging.error("Failed to save final outputs: %s", e)
        print("Error writing final results:", e)

if __name__ == "__main__":
    main()
