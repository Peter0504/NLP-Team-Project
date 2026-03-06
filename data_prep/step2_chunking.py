# -*- coding: utf-8 -*-
"""
StockSense - Step 2: Text Cleaning, Speaker Detection & Chunking
Pipeline: Strip headers/footers -> Detect speaker turns -> Segment -> Store in Postgres
"""

import re
import pickle
import random
import pandas as pd
import spacy
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from tqdm import tqdm
import kagglehub
import shutil

# -----------------------------
# 1. Setup & Initialization
# -----------------------------
print("Loading spaCy model (en_core_web_sm)...")
try:
    nlp = spacy.load('en_core_web_sm')
    print('spaCy model loaded:', nlp.meta['name'])
except OSError:
    print("❌ ERROR: spaCy model not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
    exit(1)

# Use relative local paths instead of Colab's /content/
BASE_DIR = Path(".")
RAW_DIR = BASE_DIR / "data" / "raw" / "kaggle"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Load Raw Data
# -----------------------------
data_file = RAW_DIR / 'motley-fool-data.pkl'

# Download if it doesn't exist locally yet
if not data_file.exists():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download('tpotterer/motley-fool-scraped-earnings-call-transcripts')
    for f in Path(path).rglob('*'):
        if f.is_file():
            shutil.copy2(f, RAW_DIR / f.name)

print(f"Loading data from {data_file}...")
with open(data_file, 'rb') as fh:
    df = pickle.load(fh)

print(f"Loaded {len(df):,} transcripts | columns: {list(df.columns)}")

# -----------------------------
# 3. Strip Headers / Footers
# -----------------------------
print("Cleaning headers, footers, and noise...")

HEADER_RE = re.compile(r'^.*?(?:Prepared Remarks:|PREPARED REMARKS:)', re.DOTALL)
FOOTER_RE = re.compile(
    r'(?:'
    r'10 stocks we like better than|'
    r'This article is a transcript|'
    r'The Motley Fool has a disclosure policy|'
    r'\*\s*The\s+\$\d+\s+Sizzle|'
    r'All earnings call transcripts'
    r').*$',
    re.DOTALL | re.IGNORECASE
)
NOISE_RE = re.compile(r'(<[^>]+>|&\w+;|\r|[ \t]{2,})')

def strip_header_footer(text: str) -> str:
    text = HEADER_RE.sub('', text, count=1).strip()
    text = FOOTER_RE.sub('', text).strip()
    text = NOISE_RE.sub(' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

df['clean_transcript'] = df['transcript'].apply(strip_header_footer)

# -----------------------------
# 4. Speaker-Turn Detection
# -----------------------------
print("Running speaker-turn detection and segmentation...")

MGMT_TITLES = re.compile(
    r'\b(CEO|CFO|COO|CTO|CMO|President|Chairman|'
    r'Chief Executive|Chief Financial|Chief Operating|'
    r'Director|Vice President|VP\b|SVP|EVP|'
    r'Founder|Head of)\b',
    re.IGNORECASE
)
ANALYST_RE  = re.compile(r'\b(Analyst|Research|Equity|Securities|Capital|Advisors)\b', re.IGNORECASE)
OPERATOR_RE = re.compile(r'^\s*Operator\s*$', re.IGNORECASE)
SECTION_RE  = re.compile(
    r'^\s*(Prepared Remarks|Questions and Answers|Q(?:uestions)?\s*[&and]+\s*A(?:nswers)?|'
    r'Operator Instructions)\s*$',
    re.IGNORECASE
)

def classify_role(line: str) -> str:
    if OPERATOR_RE.search(line): return 'Operator'
    if MGMT_TITLES.search(line): return 'Management'
    if ANALYST_RE.search(line):  return 'Analyst'
    return 'Unknown'

def detect_speaker(line: str):
    stripped = line.strip()
    if not stripped: return None
    if SECTION_RE.match(stripped): return None
    
    words = stripped.split()
    if len(words) > 15: return None

    role = classify_role(stripped)
    if role != 'Unknown':
        name_match = re.match(r'^([A-Z][\w\s\-\']{1,40}?)(?:\s*[\-–,]|\s+(?:CEO|CFO|COO|CTO|Analyst|VP))', stripped)
        name = name_match.group(1).strip() if name_match else stripped
        return (name, role)

    doc = nlp(stripped)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return (ent.text, 'Unknown')

    if re.match(r'^Operator$', stripped, re.IGNORECASE):
        return ('Operator', 'Operator')
    return None

def segment_transcript(call_id: int, text: str) -> list[dict]:
    chunks = []
    current_speaker = None
    current_role    = None
    buffer = []

    def flush():
        content = ' '.join(buffer).strip()
        content = re.sub(r'\s{2,}', ' ', content)
        if current_speaker and content:
            chunks.append({
                'call_id': call_id,
                'speaker': current_speaker,
                'role':    current_role,
                'text':    content,
            })

    for line in text.split('\n'):
        result = detect_speaker(line)
        if result:
            flush()
            buffer = []
            current_speaker, current_role = result
        else:
            buffer.append(line)

    flush() 
    return chunks

all_chunks = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc='Segmenting'):
    chunks = segment_transcript(call_id=idx, text=row['clean_transcript'])
    all_chunks.extend(chunks)

chunks_df = pd.DataFrame(all_chunks)
chunks_df.insert(0, 'chunk_id', range(len(chunks_df)))

print(f'\nTotal chunks produced : {len(chunks_df):,}')
print(f'Role distribution:\n{chunks_df["role"].value_counts()}')

# -----------------------------
# 5. Store chunks in Postgres
# -----------------------------
# WARNING: Update your password below if you have set up a Postgres Database!
ENABLE_POSTGRES = False # Change to True if your database is running
DB_CONFIG = dict(
    host   = 'localhost',
    port   = 5432,
    dbname = 'earnings_db', # Note: Step 1 used 'stocksense'. Ensure these match your actual DB name!
    user   = 'postgres',
    password = 'yourpassword', 
)

if ENABLE_POSTGRES:
    print("\n[Postgres] Connecting to database and inserting chunks...")
    CREATE_CALLS_TABLE = """
    CREATE TABLE IF NOT EXISTS earnings_calls (
        call_id    SERIAL PRIMARY KEY,
        ticker     TEXT,
        quarter    TEXT,
        date       TEXT,
        exchange   TEXT,
        transcript TEXT
    );
    """

    CREATE_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS transcript_chunks (
        chunk_id   BIGINT PRIMARY KEY,
        call_id    INT    REFERENCES earnings_calls(call_id),
        speaker    TEXT   NOT NULL,
        role       TEXT   NOT NULL,
        text       TEXT   NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_call_id ON transcript_chunks(call_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_role    ON transcript_chunks(role);
    """
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_CALLS_TABLE)

                calls_rows = [
                    (int(i), row['ticker'], row['q'], row['date'], row['exchange'], row['transcript'])
                    for i, row in df.iterrows()
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO earnings_calls(call_id, ticker, quarter, date, exchange, transcript)
                    VALUES %s
                    ON CONFLICT (call_id) DO NOTHING;
                    """,
                    calls_rows,
                    page_size=500
                )

                cur.execute(CREATE_CHUNKS_TABLE)

                chunk_rows = [
                    (int(r['chunk_id']), int(r['call_id']), r['speaker'], r['role'], r['text'])
                    for _, r in chunks_df.iterrows()
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO transcript_chunks(chunk_id, call_id, speaker, role, text)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO NOTHING;
                    """,
                    chunk_rows,
                    page_size=1000
                )
                conn.commit()

        print('✅ Both tables created and populated.')

        # -----------------------------
        # 6. DB-side verification
        # -----------------------------
        chunk_counts = chunks_df.groupby('call_id').size().reset_index(name='total_chunks')
        spot_ids = random.sample(list(chunk_counts['call_id']), min(10, len(chunk_counts)))

        with psycopg2.connect(**DB_CONFIG) as conn:
            query = """
            SELECT
                ec.call_id,
                ec.ticker,
                ec.quarter,
                COUNT(tc.chunk_id)                                   AS total_chunks,
                COUNT(*) FILTER (WHERE tc.role = 'Management')       AS mgmt,
                COUNT(*) FILTER (WHERE tc.role = 'Analyst')          AS analyst,
                COUNT(*) FILTER (WHERE tc.role = 'Operator')         AS operator
            FROM earnings_calls  ec
            JOIN transcript_chunks tc USING (call_id)
            WHERE ec.call_id = ANY(%s)
            GROUP BY ec.call_id, ec.ticker, ec.quarter
            ORDER BY ec.call_id;
            """
            db_spot = pd.read_sql(query, conn, params=(spot_ids,))

        print('\n=== DB spot-check (10 calls) ===')
        print(db_spot.to_string(index=False))

    except Exception as e:
        print(f"❌ Postgres operation failed: {e}")
else:
    print("\n⚠️ Skipping Database Insertion. (Set ENABLE_POSTGRES = True to enable)")

# --- THE MISSING FIX: Define the folder and save the data locally ---
PROC_DIR = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True) # Ensures the folder actually exists

chunks_path = PROC_DIR / "transcript_chunks.parquet"
chunks_df.to_parquet(chunks_path, index=False)
print(f"✅ Saved chunked data to {chunks_path}")

print("\nStep 2 Complete!")