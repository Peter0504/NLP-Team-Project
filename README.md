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
```

### 2. Download the spaCy Model
You must download the English language model for the NER pipeline to function:
```Bash
python -m spacy download en_core_web_sm
```

### 3. Environment Variables
You will need an active OpenAI API Key to generate embeddings and run the GPT-4 LLM.
Set your API key in your environment, or add it directly to app/rag_engine.py:
```Python
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
```

### 4. Run the Pipeline
To process the data, run the orchestrator script. (Note: If you have already processed the .parquet files, you can skip this step).
```Bash
python run_all.py
```

### 5. Start the API Server
Once the data is indexed and saved to the data/processed/ directory, launch the FastAPI server:
```Bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to http://127.0.0.1:8000/docs in your web browser to access the interactive Swagger UI and test the /query endpoint!
