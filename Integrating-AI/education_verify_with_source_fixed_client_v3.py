# education_verify_with_source_fixed_client_v3.py
"""
Education lookup + source verification v3
- Single genai.Client reuse
- Login-walled domains (LinkedIn) => NO-FETCH (Verified = N/A, confidence = medium)
- Trusted/firms directories => ALWAYS FETCH + fuzzy matching (more tolerant)
- Saves raw model responses and fetched HTML for audit
- Produces audit_unverified.csv for rows where Verified != Yes
- Checkpointing to resume runs
"""

import os
import sys
import time
import re
import logging
import random
from urllib.parse import urlparse
from dotenv import load_dotenv
import difflib
import csv

import pandas as pd
import requests

# try import genai
try:
    from google import genai
except Exception:
    genai = None

# ---------------- Config ----------------
INPUT_FILE = "input_data.xlsx"
OUTPUT_FILE = "output_with_education_verified_v3.csv"
AUDIT_FILE = "audit_unverified_v3.csv"
CHECKPOINT_FILE = "education_checkpoint_v3.txt"
RAW_RESP_DIR = "raw_responses_verified_v3"
RAW_PAGES_DIR = "raw_pages_v3"
LOG_FILE = "education_verify_v3.log"

MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.0

BATCH_SIZE = 1           # keep 1 for best accuracy; increase to speed
MAX_ATTEMPTS = 6
BACKOFF_BASE = 4
BACKOFF_MAX = 180
SLEEP_SECONDS = 1.5

NAME_COL = "Name"
ORG_COL = "Organization/Law Firm Name"

COL_EDU = "Education (AI Found)"
COL_SRC = "Education Source (AI Found)"
COL_VER = "Source Verified (Yes/No/N/A/Trusted)"
COL_YEAR = "Education Year (AI Found)"
COL_CONF = "AI_Confidence"

HTTP_TIMEOUT = 12
HTTP_HEADERS = {"User-Agent": "education-verifier/1.0 (+https://example.com)"}

# Domains to never fetch (login-wall / blocking) -> mark NO_FETCH
NO_FETCH_DOMAINS = {
    "linkedin.com", "www.linkedin.com", "linkedin.cn"
}

# Domains that are authoritative but we WILL fetch and verify (firm bios, martindale, justia etc.)
FETCH_AND_VERIFY_DOMAINS = {
    "martindale.com", "www.martindale.com",
    "justia.com", "www.justia.com",
    "kirkland.com", "www.kirkland.com",
    "mofo.com", "www.mofo.com",
    "dowlingaaron.com", "www.dowlingaaron.com",
    "knobbe.com", "www.knobbe.com",
    # Add more law-firm domains as desired
}

# fallback trusted assume list (if you prefer trusting without fetch add domains here)
TRUSTED_ASSUME_DOMAINS = {
    # (empty by default; prefer fetch for important domains)
}

# similarity threshold (0..1) for fuzzy match; lower -> more permissive
FUZZY_THRESHOLD = 0.22

# ---------------- Logging ----------------
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)

load_dotenv()

# ---------------- Helpers ----------------
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

def safe_get_response_text(response):
    if response is None:
        return ""
    try:
        if hasattr(response, "text"):
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
        else:
            continue
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
            if re.search(r'year\s*[:\-]\s*n/?a', p, flags=re.IGNORECASE):
                return "N/A"
    m = re.search(r'J\.?D\.?.{0,12}?(\b(19|20)\d{2}\b)', full_text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return "N/A"

def domain_of(url):
    if not url or url.strip().upper() == "N/A":
        return None
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc
    except Exception:
        return None

def fuzzy_similarity(a, b):
    "Return sequence match ratio between two strings (0..1)."
    if not a or not b:
        return 0.0
    try:
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0

def verify_page_fuzzy(url, expected_text):
    """
    Fetch url, save HTML, and attempt token + fuzzy match.
    Returns:
      True  -> verified
      False -> fetched but not matched
      'NO_FETCH' -> domain in NO_FETCH_DOMAINS
      'N/A' -> no expected_text
    """
    if not url or url.strip().upper() == "N/A":
        return "N/A"
    dom = domain_of(url)
    if dom is None:
        return False
    if dom in NO_FETCH_DOMAINS:
        return "NO_FETCH"
    # if domain in TRUSTED_ASSUME_DOMAINS, we could assume trust, but prefer fetch for verification list
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            url_try = "https://" + url.lstrip("/")
        else:
            url_try = url
        r = requests.get(url_try, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200 and url_try.startswith("https://"):
            try:
                url_try2 = "http://" + url_try[len("https://"):]
                r2 = requests.get(url_try2, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
                if r2.status_code == 200:
                    page_text = r2.text.lower()
                    page_html = r2.text
                    used_url = url_try2
                else:
                    return False
            except Exception:
                return False
        else:
            page_text = r.text.lower()
            page_html = r.text
            used_url = url_try
    except Exception as e:
        logging.debug("Fetch failed for %s: %s", url, str(e))
        return False

    # save HTML for audit (filename sanitized)
    try:
        fname = re.sub(r'[^0-9a-zA-Z_\-\.]', '_', dom) + "_" + str(int(time.time())) + ".html"
        path = os.path.join(RAW_PAGES_DIR, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(page_html)
        logging.info("Saved fetched page to %s", path)
    except Exception as e:
        logging.debug("Failed to save fetched page HTML: %s", e)

    if not expected_text or expected_text.upper() == "N/A":
        return "N/A"

    # token presence check (conservative)
    tokens = []
    raw = re.sub(r'[\(\)\[\]\|]', ' ', expected_text)
    cand = re.split(r'[;,\-–—,]', raw)
    for c in cand:
        t = c.strip()
        # ignore tiny tokens
        if len(t) >= 4:
            tokens.append(t.lower())

    # if tokens absent, fallback to fuzzy compare expected_text vs first part of page
    if not tokens:
        # use the first N chars of page_text for fuzzy
        sample = page_text[:6000]
        sim = fuzzy_similarity(expected_text.lower(), sample)
        return sim >= FUZZY_THRESHOLD

    # If all token words appear in page_text (loose), accept True
    all_present = True
    for tk in tokens:
        # split token into words of length>2 and ensure presence of at least two words
        words = [w for w in re.findall(r'\w+', tk) if len(w) > 2]
        if not words:
            continue
        if not all(w in page_text for w in words):
            all_present = False
            break
    if all_present:
        return True

    # otherwise compute fuzzy similarity between expected_text and page head
    sample = page_text[:8000]
    sim = fuzzy_similarity(expected_text.lower(), sample)
    return sim >= FUZZY_THRESHOLD

# ---------------- genai client init & call ----------------
def init_genai_client():
    if genai is None:
        return None, "google-genai SDK not installed (pip install google-genai)"
    api_key = os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            logging.info("Initialized genai client with API key from environment.")
            return client, None
        except Exception as e:
            logging.error("genai.Client(init with api_key) failed: %s", e, exc_info=True)
            return None, f"genai.Client init failed with API key: {e}"
    else:
        try:
            client = genai.Client()
            logging.info("Initialized genai client with default settings (no explicit api_key).")
            return client, None
        except Exception as e:
            logging.error("genai.Client default init failed: %s", e, exc_info=True)
            msg = ("Missing API key/environment. Set GENAI_API_KEY or GOOGLE_API_KEY env var or configure Vertex settings.")
            return None, msg

def call_with_retries_shared_client(client, prompt, max_attempts=MAX_ATTEMPTS):
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
            logging.warning("Attempt %d/%d failed: %s", attempt, max_attempts, msg.splitlines()[0])
            if any(x in msg.lower() for x in ("503", "service unavailable", "429", "rate limit", "quota", "unavailable", "internal")):
                if attempt >= max_attempts:
                    return False, msg
                backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))) + random.uniform(0, 2.0)
                logging.info("Transient error: backing off %.1f s and retrying...", backoff)
                time.sleep(backoff)
                continue
            else:
                logging.error("Non-transient genai error: %s", msg)
                return False, msg
    return False, "Unknown failure after retries"

# ---------------- Main ----------------
def main():
    ensure_dirs()
    client, err = init_genai_client()
    if client is None:
        logging.error("genai client not initialized: %s", err)
        print("genai client not initialized:", err)
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
    logging.info("Resuming from row (0-based): %d", start)
    processed = start

    try:
        for batch_start in range(start, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch_df = df.iloc[batch_start:batch_end].copy()
            lines = []
            index_map = []
            for j, (orig_idx, row) in enumerate(batch_df.iterrows(), start=1):
                name = str(row.get(NAME_COL, "")).strip()
                org = str(row.get(ORG_COL, "")).strip() if ORG_COL in df.columns else ""
                lines.append(f"{j}. {name} @ {org}")
                index_map.append(orig_idx)

            prompt = ("You are an expert researcher with access to public directories up to 2025.\n"
                      "For each person below, list all verified education entries and include a short source URL.\n"
                      "Format EXACTLY:\n1. <entries separated by ';'> (source: <short url or N/A>)\n"
                      "If nothing credible is found: N/A (source: N/A)\nPeople to research:\n")
            prompt += "\n".join(lines)

            logging.info("Querying rows %d..%d", batch_start, batch_end - 1)
            ok, resp_text = call_with_retries_shared_client(client, prompt)
            timestamp = int(time.time())
            if ok:
                rawfile = os.path.join(RAW_RESP_DIR, f"resp_{batch_start}_{timestamp}.txt")
                try:
                    with open(rawfile, "w", encoding="utf-8") as fh:
                        fh.write(resp_text)
                    logging.info("Saved raw response to %s", rawfile)
                except Exception as e:
                    logging.warning("Failed to save raw response: %s", e)

                parsed = parse_numbered_list(resp_text)
                for j, orig_idx in enumerate(index_map, start=1):
                    key = str(j)
                    raw_val = parsed.get(key, "N/A")
                    edu_text, src = extract_edu_and_source(raw_val)
                    year = extract_jd_year(edu_text)

                    dom = domain_of(src)
                    verified = "N/A"
                    confidence = "low"

                    # Policy:
                    if not src or src.upper() == "N/A":
                        verified = "N/A"
                        confidence = "low" if edu_text.upper() == "N/A" else "medium"
                    else:
                        if dom in NO_FETCH_DOMAINS:
                            verified = "N/A"
                            confidence = "medium"
                            logging.info("Domain %s is NO-FETCH; skipping fetch", dom)
                        elif dom in FETCH_AND_VERIFY_DOMAINS:
                            v = verify_page_fuzzy(src, edu_text)
                            if v == True:
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
                        elif dom in TRUSTED_ASSUME_DOMAINS:
                            verified = "Trusted"
                            confidence = "high"
                        else:
                            # attempt fetch for unknown domains too (best-effort)
                            v = verify_page_fuzzy(src, edu_text)
                            if v == True:
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

                    preview = (edu_text[:160] + "...") if len(edu_text) > 160 else edu_text
                    print(f"{batch_start + j - 1:6d}: {df.at[orig_idx, NAME_COL]:40.40s} -> {preview:60s} | src: {src:30s} | ver: {verified} | conf: {confidence}")

                processed = batch_end
                save_checkpoint(processed)
                try:
                    df.to_csv(OUTPUT_FILE, index=False)
                    logging.info("Intermediate save to %s", OUTPUT_FILE)
                except Exception as e:
                    logging.warning("Failed intermediate save: %s", e)
                time.sleep(SLEEP_SECONDS + random.uniform(0, 1.0))
            else:
                logging.error("Persistent genai failure for batch %d..%d : %s", batch_start, batch_end - 1, resp_text)
                for orig_idx in index_map:
                    df.at[orig_idx, COL_EDU] = "N/A"
                    df.at[orig_idx, COL_SRC] = "N/A"
                    df.at[orig_idx, COL_VER] = "N/A"
                    df.at[orig_idx, COL_YEAR] = "N/A"
                    df.at[orig_idx, COL_CONF] = "low"
                processed = batch_end
                save_checkpoint(processed)
                try:
                    df.to_csv(OUTPUT_FILE, index=False)
                except Exception:
                    pass
                backoff = min(BACKOFF_MAX, 30 + random.uniform(0, 30))
                logging.info("Backing off %.1f s after persistent failure", backoff)
                time.sleep(backoff)

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Saving progress up to %d and exiting.", processed)
        try:
            df.to_csv(OUTPUT_FILE, index=False)
        except Exception:
            pass
        print("\nInterrupted by user. Progress saved.")
    finally:
        try:
            if client is not None and hasattr(client, "close") and callable(client.close):
                client.close()
                logging.info("Closed genai client.")
        except Exception as e:
            logging.debug("Error closing client: %s", e)

    # final save and audit CSV for unverified rows
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info("Final output saved to %s", OUTPUT_FILE)
        # write audit file for rows where Verified != Yes
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
