from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import lru_cache
import time

# Import our custom engine
from app.core.retrieval import HybridRetriever

app = FastAPI(title="TechDocs RAG API")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine
retriever = HybridRetriever()

# Mock Data (Simulating 15k+ pages)
mock_docs = [
    "To reset the API key, go to settings and click revoke.",
    "The 404 error usually occurs when the endpoint URL is misspelled.",
    "Python 3.11 introduces significant speed improvements.",
    "Use the --verbose flag to see detailed logs during deployment.",
    "The database connection pool size should be set to 20 for production.",
    # ... In reality, you would load this from a database or files
]

@app.on_event("startup")
async def startup_event():
    # Load data on startup
    retriever.ingest_documents(mock_docs)

# Request Model
class QueryRequest(BaseModel):
    question: str

# 1. Intelligent Caching (LRU Cache)
# Reduces API costs by not re-processing identical queries
@lru_cache(maxsize=1000)
def cached_search(question: str):
    # Hybrid Search
    candidates = retriever.search(question, top_k=10)
    # Reranking
    final_docs = retriever.rerank(question, candidates, top_n=3)
    return final_docs

@app.post("/query")
async def query_documentation(request: QueryRequest):
    start_time = time.time()
    
    # Check cache wrapper
    context_docs = cached_search(request.question)
    
    # 2. Context Windowing
    # Limit context size to prevent exceeding token limits (cost reduction)
    context_text = "\n\n".join(context_docs)
    MAX_CONTEXT_CHARS = 2000 
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "...(truncated)"

    # Mock LLM Generation (Replace with OpenAI/Anthropic call)
    answer = f"Based on the docs: {context_text[:100]}... [Generated Answer Here]"
    
    process_time = (time.time() - start_time) * 1000
    
    return {
        "answer": answer,
        "context": context_docs,
        "latency_ms": round(process_time, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)