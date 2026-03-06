# StockSense: Financial Earnings Call RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system designed to process, enrich, and query a massive dataset of financial earnings call transcripts. 

This project transforms over 18,000 raw financial transcripts into a structured, searchable local vector database, and serves a FastAPI backend that allows users to ask complex financial questions using GPT-4, strictly grounded in the provided data.

## 🧠 System Architecture

This pipeline is broken down into four distinct stages:

1. **Data Ingestion & Segmenting:** - Cleans unstructured transcript text.
   - Applies speaker-turn detection to segment text into discrete chunks, categorizing speakers by role (Management vs. Analyst).
2. **NLP Enrichment:** - **Sentiment Analysis:** Utilizes `FinancialBERT` to score the emotional tone of each chunk.
   - **Named Entity Recognition (NER):** Utilizes `spaCy` to extract key financial organizations, tickers, and monetary metrics.
   - **Topic Modeling:** Utilizes `Gensim` (LDA) to categorize overarching themes.
3. **Hybrid Search Indexing:** - Generates high-dimensional vectors using OpenAI's `text-embedding-3-small`.
   - Builds a local `FAISS` vector database for semantic similarity search.
   - Builds a `BM25` index for exact keyword retrieval.
4. **FastAPI RAG Engine:** - Orchestrates a custom hybrid retriever combining dense (FAISS) and sparse (BM25) results.
   - Uses LangChain to feed the retrieved context to GPT-4.
   - Exposes a REST API endpoint for querying.

## 🚀 Quickstart Guide

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Clone this repository and install the required packages:
```bash
pip install -r requirements.txt
