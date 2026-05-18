# --- CRITICAL HUGGING FACE CHROMADB FLIGHT PATCH ---
# Must execute before any vector stores or langchain modules get imported
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# --------------------------------------------------

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List

# Standard root folder setup
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

# Importing pipeline engines
from src.pipeline import GeminiRAG
from src.ingestion import IngestionPipeline

load_dotenv()

app = FastAPI(
    title="RAG Research API",
    version="1.0.0"
)

# Initialize systems safely after the sqlite3 patch is live
rag_system = GeminiRAG()
ingestor = IngestionPipeline()

# Safe scratch directory isolated from the live workspace hot-reloader
UPLOAD_DIR = Path("/tmp/rag_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/")
async def root_status():
    return {"status": "online", "message": "RAG Research API is running."}

@app.post("/ask", response_model=QueryResponse)
async def ask_rag(request: QueryRequest):
    try:
        relevant_docs = rag_system.retrieve_and_rerank(request.prompt)
        answer = rag_system.generate(request.prompt, relevant_docs)
        sources = [doc.page_content for doc in relevant_docs]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    temp_path = UPLOAD_DIR / file.filename
    try:
        # 1. Stream incoming stream payload into OS temporary memory
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Run vector indexing safely inside transparent exception catches
        try:
            print(f"Starting vector ingestion pipeline for: {file.filename}")
            ingestor.run(str(temp_path))
            print(f"Successfully processed and indexed: {file.filename}")
            return {"status": "success", "filename": file.filename}
        except Exception as ingest_error:
            import traceback
            print(f"💥 Ingestion Pipeline Failure: {str(ingest_error)}")
            traceback.print_exc()
            
            # Bubble clean debug info to frontend UI terminal
            raise HTTPException(
                status_code=500, 
                detail=f"Ingestion Pipeline Error: {str(ingest_error)}. Verify secret tokens and Chroma space."
            )
            
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=f"File Upload Core Error: {str(e)}")
        raise e
    finally:
        # 3. Always clear physical footprint inside the ephemeral directory
        if temp_path.exists():
            os.remove(temp_path)

@app.post("/run-benchmark")
async def trigger_benchmark(background_tasks: BackgroundTasks):
    """
    Asynchronously fires background RAGAS evaluating calculations.
    """
    from tests.custom_eval import run_evaluation
    
    METRICS_FILE_PATH = "/app/shared/metrics.json"
    
    def run_and_save():
        try:
            print("Starting background RAGAS evaluation...")
            results = run_evaluation()
            results['last_run'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            os.makedirs(os.path.dirname(METRICS_FILE_PATH), exist_ok=True)
            
            with open(METRICS_FILE_PATH, "w") as f:
                json.dump(results, f)
            print(f"Benchmark completed successfully! Saved to {METRICS_FILE_PATH}")
            
        except Exception as e:
            import traceback
            print(f"💥 Benchmark Background Error: {e}")
            traceback.print_exc()

    background_tasks.add_task(run_and_save)
    return {"status": "started", "message": "Benchmark is running in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_final:app", host="0.0.0.0", port=8002, reload=False)