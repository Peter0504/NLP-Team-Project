# -*- coding: utf-8 -*-
import os
import pickle
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---> THE MISSING PIECE <---
os.environ["OPENAI_API_KEY"] = "YOUR OPEN AI API KEY HERE"

# Robustly point to the processed data folder
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

# 1. Load OpenAI Embeddings & FAISS Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    str(PROC_DIR / "faiss_index"), 
    embeddings, 
    allow_dangerous_deserialization=True
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Load BM25 Keyword Retriever
with open(PROC_DIR / "bm25_retriever.pkl", "rb") as f:
    bm25_retriever = pickle.load(f)
bm25_retriever.k = 3

# 3. Create Custom Hybrid Retriever
def custom_hybrid_search(query: str):
    """Combines BM25 and FAISS results and removes duplicates."""
    vec_docs = vector_retriever.invoke(query)
    kw_docs = bm25_retriever.invoke(query)
    
    unique_docs = {}
    for doc in vec_docs + kw_docs:
        chunk_id = doc.metadata.get('chunk_id')
        if chunk_id not in unique_docs:
            unique_docs[chunk_id] = doc
            
    return list(unique_docs.values())

hybrid_retriever = RunnableLambda(custom_hybrid_search)

# 4. Define LLM and Prompt
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt_template = """
You are a financial analyst assistant. Use the following retrieved earnings call transcripts to answer the question. 
Assess your confidence in the retrieved context. If the context does not contain the answer, output exactly: 
"LOW_CONFIDENCE: I cannot answer this based on the provided transcripts."

Retrieved Context:
{context}

Question: {question}

Answer with detailed reasoning. Append a list of source 'chunk_ids' and 'speakers' at the end of your response.
"""

prompt = PromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(f"[Chunk ID: {doc.metadata.get('chunk_id')} | Speaker: {doc.metadata.get('speaker')}]\n{doc.page_content}" for doc in docs)

# 5. Build the RAG Chain
rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)