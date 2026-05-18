import os
import shutil
import sys
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

# Importing your logic
from src.pipeline import GeminiRAG
from src.ingestion import IngestionPipeline

load_dotenv()

app = FastAPI(
    title="RAG Research API",
    version="1.0.0"
)

# Initialize the systems
rag_system = GeminiRAG()
ingestor = IngestionPipeline()

# FIX: Use Linux system /tmp space so creating/deleting temporary files 
# doesn't trigger Uvicorn/WatchFiles tracking to reload the workspace.
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
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        ingestor.run(str(temp_path))
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            os.remove(temp_path)

@app.post("/run-benchmark")
async def trigger_benchmark(background_tasks: BackgroundTasks):
    """
    Triggers the evaluation logic. 
    Make sure tests/custom_eval.py exists!
    """
    from tests.custom_eval import run_evaluation
    
    # Define the exact same absolute path your frontend is reading from
    METRICS_FILE_PATH = "/app/shared/metrics.json"
    
    def run_and_save():
        try:
            print("Starting background RAGAS evaluation...")
            results = run_evaluation()
            results['last_run'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Ensure the /app/shared parent directory exists
            os.makedirs(os.path.dirname(METRICS_FILE_PATH), exist_ok=True)
            
            # Save directly to the shared volume directory path
            with open(METRICS_FILE_PATH, "w") as f:
                json.dump(results, f)
            print(f"Benchmark completed successfully! Saved to {METRICS_FILE_PATH}")
            
        except Exception as e:
            # This logs the specific failure traceback directly into Hugging Face logs
            import traceback
            print(f"💥 Benchmark Background Error: {e}")
            traceback.print_exc()

    background_tasks.add_task(run_and_save)
    return {"status": "started", "message": "Benchmark is running in background"}

if __name__ == "__main__":
    import uvicorn
    # FIX: Explicitly enforce reload=False to ensure the backend process isn't killed mid-upload.
    uvicorn.run("main_final:app", host="0.0.0.0", port=8002, reload=False)