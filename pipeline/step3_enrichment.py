# -*- coding: utf-8 -*-
import pandas as pd
import spacy
from transformers import pipeline
from gensim import corpora, models
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(".")
PROC_DIR = BASE_DIR / "data" / "processed"
input_file = PROC_DIR / "transcript_chunks.parquet"

print(f"Loading chunked data from {input_file}...")
try:
    chunks_df = pd.read_parquet(input_file)
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {input_file}.")
    exit(1)

# Sampling 100 so your computer doesn't run for 48 hours
SAMPLE_SIZE = 100
print(f"⚠️ SAMPLING DATA: Taking the first {SAMPLE_SIZE} chunks for NLP processing...")
chunks_df = chunks_df.head(SAMPLE_SIZE).copy()

print(" -> Loading FinancialBERT...")
sentiment_pipe = pipeline("sentiment-analysis", model="ahmedrachid/FinancialBERT-Sentiment-Analysis", truncation=True, max_length=512)

print(" -> Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

def get_sentiment(text):
    try:
        res = sentiment_pipe(str(text)[:512])[0]
        return res['label'], res['score']
    except Exception:
        return "Neutral", 0.0

def extract_entities(text):
    doc = nlp(str(text))
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'MONEY', 'PERCENT']]
    return ", ".join(list(set(entities)))

print(" -> Running NLP (Sentiment & NER)...")
sentiments = chunks_df['text'].apply(get_sentiment)
chunks_df['sentiment'] = [s[0] for s in sentiments]
chunks_df['sentiment_score'] = [s[1] for s in sentiments]
chunks_df['entities'] = chunks_df['text'].apply(extract_entities)

print(" -> Running Topic Modeling (LDA)...")
texts = [[w for w in str(doc).lower().split() if len(w) > 3] for doc in chunks_df['text']]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=5)

def get_topic(text):
    bow = dictionary.doc2bow(str(text).lower().split())
    topics = lda_model.get_document_topics(bow)
    if topics:
        return f"Topic_{max(topics, key=lambda item: item[1])[0]}"
    return "Unknown"

chunks_df['dominant_topic'] = chunks_df['text'].apply(get_topic)

output_file = PROC_DIR / "enriched_chunks.parquet"
chunks_df.to_parquet(output_file, index=False)
print(f"✅ Step 3 Complete! Saved to {output_file}")