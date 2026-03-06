# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_engine import rag_chain

app = FastAPI(title="StockSense RAG API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the StockSense API! The server is running."}

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the hybrid RAG system."""
    print(f"Received question: {request.question}")
    
    try:
        # Invoke the LangChain RAG pipeline
        response = rag_chain.invoke(request.question)
        
        # Fallback/Mitigation Check
        if isinstance(response, str) and "LOW_CONFIDENCE" in response:
            return {
                "answer": "I do not have enough specific evidence in the transcripts to answer this."
                # Removed raw_response to prevent information disclosure
            }
            
        return {"answer": response}
        
    except Exception as e:
        # Log the actual error internally so you can debug it
        print(f"Error during RAG invocation: {e}")
        # Return a generic, safe error to the user
        return {"answer": "An internal error occurred while processing your request."}