# -*- coding: utf-8 -*-
"""
StockSense - Data Acquisition & Ingestion
"""

import kagglehub
import pandas as pd
import re
import hashlib
import numpy as np
import json
import filetype
import warnings
import shutil
from datetime import datetime, timezone
from pathlib import Path

# -----------------------------
# Config & Local Paths
# -----------------------------
DATASET_ID = "tpotterer/motley-fool-scraped-earnings-call-transcripts"

# Use relative local paths instead of Google Colab's /content/ path
BASE_DIR = Path(".")
RAW_DIR = BASE_DIR / "data" / "raw" / "kaggle"
META_DIR = BASE_DIR / "data" / "metadata"
PROC_DIR = BASE_DIR / "data" / "processed"
CACHE_DIR = BASE_DIR / "cache"

for p in [RAW_DIR, META_DIR, PROC_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Set to True ONLY if you have a local PostgreSQL server running with these credentials
ENABLE_POSTGRES = False  
PG_DSN = "host=localhost dbname=stocksense user=postgres password=yourpassword port=5432"

# -----------------------------
# 1. Download & Load Data
# -----------------------------
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download(DATASET_ID)
print("kagglehub path:", path)

# Copy files to raw landing zone
src = Path(path)
for f in src.rglob("*"):
    if f.is_file():
        dst = RAW_DIR / f.name
        shutil.copy2(f, dst)

data_file = RAW_DIR / "motley-fool-data.pkl"
print(f"Loading data from: {data_file}")
df_raw = pd.read_pickle(data_file)

print("Shape:", df_raw.shape)

# -------- Column mapping --------
COLMAP = {
    "ticker": "ticker",
    "call_date": "date",
    "quarter": "q",
    "exchange": "exchange",
    "raw_text": "transcript",
}

# -----------------------------
# 2. Helpers
# -----------------------------
def _scalarize(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0: return None
        return x[0]
    return x

def normalize_quarter(x):
    x = _scalarize(x)
    if x is None or pd.isna(x): return None
    s = str(x).strip().upper()
    m = re.search(r"(Q[1-4])", s)
    return m.group(1) if m else None

def extract_year(x):
    x = _scalarize(x)
    if x is None or pd.isna(x): return None
    s = str(x)
    m = re.search(r"(20\d{2})", s)
    return int(m.group(1)) if m else None

def clean_date_str(s):
    s = _scalarize(s)
    if s is None or pd.isna(s): return None
    s = str(s).strip()
    s = re.sub(r"\s+ET$", "", s) 
    s = s.replace("a.m.", "AM").replace("p.m.", "PM")
    return s

def parse_call_datetime_multi(series: pd.Series) -> pd.Series:
    s = series.apply(clean_date_str)
    dt = pd.to_datetime(s, format="%b %d, %Y, %I:%M %p", errors="coerce")
    mask = dt.isna() & s.notna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(s[mask], format="%b %d, %Y", errors="coerce")
    mask = dt.isna() & s.notna()
    if mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dt.loc[mask] = pd.to_datetime(s[mask], errors="coerce")
    return dt

def stable_doc_id(ticker: str, quarter_raw: str, raw_text: str) -> str:
    base = f"{ticker}|{quarter_raw}|{raw_text}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def normalized_name(row):
    y = row["fiscal_year"] if pd.notna(row["fiscal_year"]) else "NA"
    q = row["fiscal_quarter"] if row["fiscal_quarter"] else "NA"
    return f"{row['ticker']}_{y}_{q}_{row['doc_id'][:10]}"

def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def embedded_symbol(exchange_str):
    if exchange_str is None: return None
    s = str(exchange_str)
    if ":" in s:
        return s.split(":", 1)[1].strip().upper()
    return None

# -----------------------------
# 3. Build standardized dataset
# -----------------------------
print("Cleaning and standardizing data...")
out = pd.DataFrame()
out["ticker"] = df_raw[COLMAP["ticker"]].astype(str).str.upper().str.strip()
out["quarter_raw"] = df_raw[COLMAP["quarter"]].astype(str)

parsed_dt = parse_call_datetime_multi(df_raw[COLMAP["call_date"]])
out["call_date"] = parsed_dt.dt.date
out["fiscal_quarter"] = out["quarter_raw"].apply(normalize_quarter)
out["fiscal_year"] = out["quarter_raw"].apply(extract_year).astype("Int64")
out["exchange"] = df_raw[COLMAP["exchange"]].astype(str)
out["raw_text"] = df_raw[COLMAP["raw_text"]].astype(str)
out["text_len"] = out["raw_text"].str.len()
out["source"] = "kaggle_motleyfool"
out["ingested_at"] = datetime.now(timezone.utc).isoformat()

# Filter garbled text and deduplicate
out = out[out["text_len"] >= 500].copy()
out["doc_id"] = [stable_doc_id(t, q, txt) for t, q, txt in zip(out["ticker"].values, out["quarter_raw"].values, out["raw_text"].values)]

before = len(out)
out = out.drop_duplicates(subset=["doc_id"], keep="first").copy()
dropped = before - len(out)

out["normalized_name"] = out.apply(normalized_name, axis=1)

# -----------------------------
# 4. Save outputs 
# -----------------------------
proc_path = PROC_DIR / "transcripts_clean.parquet"
meta_path = META_DIR / "transcripts_metadata.csv"

out.to_parquet(proc_path, index=False)
meta_cols = ["doc_id", "ticker", "call_date", "fiscal_year", "fiscal_quarter", "exchange", "source", "text_len", "ingested_at", "normalized_name"]
out[meta_cols].to_csv(meta_path, index=False)

# doc_index.json
doc_index = out[["doc_id", "ticker", "fiscal_year", "fiscal_quarter", "call_date", "text_len", "source"]].copy()
doc_index["call_date"] = doc_index["call_date"].astype(str)
doc_index_path = CACHE_DIR / "doc_index.json"
with open(doc_index_path, "w") as f:
    json.dump(doc_index.to_dict(orient="records"), f)

# Manifest
kind = filetype.guess(str(data_file))
manifest = {
    "dataset_id": DATASET_ID,
    "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "rows": int(len(out)),
    "dropped_exact_duplicates": int(dropped),
}
manifest_path = CACHE_DIR / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("\n--- Summary ---")
print(f"Rows processed: {len(out)}")
print(f"Dropped exact duplicates: {dropped}")
print(f"Distinct tickers: {out['ticker'].nunique()}")
print(f"Saved processed data to: {proc_path}")

# -----------------------------
# 5. Postgres ETL (Optional)
# -----------------------------
if ENABLE_POSTGRES:
    import psycopg2
    from psycopg2.extras import execute_values
    from datetime import date as _date

    print("\n[Postgres] Connecting to local Postgres database...")
    
    def make_period(y, q):
        if pd.isna(y) or not q: return None
        return f"{int(y)}{q}" 

    def to_py_date_or_none(x):
        if x is None or pd.isna(x): return None
        if isinstance(x, _date): return x
        try:
            ts = pd.to_datetime(x, errors="coerce")
            return ts.date() if not pd.isna(ts) else None
        except Exception:
            return None

    def to_timestamptz_or_none(x):
        if x is None or pd.isna(x): return None
        s = str(x).strip()
        return s if s and s.lower() != "nat" else None

    try:
        conn = psycopg2.connect(PG_DSN)
        conn.autocommit = False

        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
              doc_id TEXT PRIMARY KEY,
              ticker VARCHAR(10) NOT NULL,
              period VARCHAR(10),
              call_date DATE,
              fiscal_year INT,
              fiscal_quarter VARCHAR(2),
              exchange VARCHAR(60),
              source VARCHAR(30) NOT NULL,
              raw_text TEXT NOT NULL,
              text_len INT,
              metadata JSONB,
              ingested_at TIMESTAMPTZ
            );
            """)
        conn.commit()

        rows = []
        for r in out.itertuples(index=False):
            md = {
                "normalized_name": r.normalized_name,
                "quarter_raw": getattr(r, "quarter_raw", None),
                "exchange": r.exchange,
                "source": r.source,
            }
            rows.append((
                r.doc_id, r.ticker, make_period(r.fiscal_year, r.fiscal_quarter),
                to_py_date_or_none(r.call_date), int(r.fiscal_year) if pd.notna(r.fiscal_year) else None,
                r.fiscal_quarter, r.exchange, r.source, r.raw_text, int(r.text_len),
                json.dumps(md), to_timestamptz_or_none(r.ingested_at)
            ))

        with conn.cursor() as cur:
            execute_values(cur, """
            INSERT INTO transcripts (
              doc_id, ticker, period, call_date, fiscal_year, fiscal_quarter,
              exchange, source, raw_text, text_len, metadata, ingested_at
            ) VALUES %s
            ON CONFLICT (doc_id) DO NOTHING;
            """, rows, page_size=1000)
        conn.commit()
        conn.close()
        print("Postgres load complete.")
    except Exception as e:
        print(f"Postgres insertion failed: {e}")

print("\nStep 1 Complete!")

