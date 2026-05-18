# --- CRITICAL HUGGING FACE CHROMADB FLIGHT PATCH ---
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import json
import time
import os
import shutil
from datetime import datetime
from pathlib import Path

# Fix relative import structures if needed
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if root not in sys.path:
    sys.path.insert(0, root)

# Direct internal pipeline engine imports
from src.pipeline import GeminiRAG
from src.ingestion import IngestionPipeline
from dotenv import load_dotenv

load_dotenv()

# 1. Page Configuration
st.set_page_config(
    page_title="Transformer RAG Expert",
    page_icon="🔬",
    layout="wide"
)

# 2. Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .st-emotion-cache-1c7n2ka { max-width: 95%; } 
    .sidebar-text { font-size: 14px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

# 3. Persistent Local Paths
METRICS_FILE_PATH = "metrics.json"  # Localized path for space container persistence
UPLOAD_DIR = Path("/tmp/rag_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 4. Initialize Local Core Engines safely using st.cache_resource
@st.cache_resource
def init_rag_system():
    try:
        return GeminiRAG(), IngestionPipeline()
    except Exception as e:
        st.error(f"Failed to initialize engines: {e}. Check API keys in settings.")
        return None, None

rag_system, ingestor = init_rag_system()

# 5. Sidebar - Dashboard & Management
with st.sidebar:
    st.title("📊 Control Center")
    
    # Clean UI System Status Indicator
    if rag_system and ingestor:
        st.success("API Status: Online (Internal Engine Connected)")
    else:
        st.error("API Status: Offline (Initialization Failed)")

    st.divider()
    
    # Local Metrics Reader
    def load_metrics():
        try:
            with open(METRICS_FILE_PATH, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"faithfulness": 0, "answer_relevance": 0, "last_run": "N/A"}

    metrics = load_metrics()

    st.subheader("Performance Metrics")
    col1, col2 = st.columns([1, 1])

    # Mapping to correct keys based on your main_final.py schema
    faithfulness = metrics.get('faithfulness', 0)
    relevancy = metrics.get('answer_relevance', 0) if 'answer_relevance' in metrics else metrics.get('relevancy', 0)

    col1.metric("Faithfulness", f"{faithfulness * 100:.0f}%")
    col2.metric("Relevancy", f"{relevancy * 100:.0f}%")

    st.caption(f"Last benchmark run: {metrics.get('last_run', 'N/A')}")

    st.divider()
    
    st.subheader("🧪 Developer Tools")
    if st.button("🚀 Run System Benchmark", use_container_width=True):
        if not rag_system:
            st.error("Engines are offline.")
        else:
            with st.status("Running RAGAS Evaluation...", expanded=True) as status:
                try:
                    from tests.custom_eval import run_evaluation
                    
                    status.write("Starting RAGAS background pipeline metrics calculation...")
                    results = run_evaluation()
                    
                    if results is None or not isinstance(results, dict):
                        results = {
                            "faithfulness": 0.0,
                            "answer_relevance": 0.0,
                            "error": "Evaluation execution returned empty dataset."
                        }
                    
                    results['last_run'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    with open(METRICS_FILE_PATH, "w") as f:
                        json.dump(results, f)
                        
                    status.update(label="Benchmark Started!", state="complete")
                    st.toast("Evaluation completed and saved!", icon="✅")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    status.update(label="Benchmark Failed", state="error")
                    st.error(f"Error running benchmark script: {e}")

    st.divider()

    # Knowledge Management (Direct Ingestion Engine Injection)
    st.subheader("📥 Knowledge Management")
    st.markdown('<p class="sidebar-text">Add new research papers to the Vector Store.</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
    
    if uploaded_file and ingestor:
        if st.button("Process & Index Document", use_container_width=True):
            with st.spinner("Brain in progress... indexing..."):
                temp_path = UPLOAD_DIR / uploaded_file.name
                try:
                    # Write stream data to temporary container execution directory
                    with temp_path.open("wb") as buffer:
                        buffer.write(uploaded_file.getvalue())
                    
                    # Direct processing
                    ingestor.run(str(temp_path))
                    
                    st.balloons()
                    st.success(f"Indexed: {uploaded_file.name}")
                except Exception as ingest_error:
                    st.error(f"Ingestion Pipeline Error: {ingest_error}")
                finally:
                    if temp_path.exists():
                        os.remove(temp_path)

    st.divider()
    if st.button("Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 6. Main Chat UI Header
st.title("🔬 Research Assistant: Transformer Architecture")
st.info("Ask complex questions about Multi-Head Attention, Positional Encodings, or any uploaded papers.")

# 7. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Evidence (Retrieved Context)"):
                for idx, src in enumerate(message["sources"]):
                    st.caption(f"Source Chunk {idx+1}:")
                    st.write(src)
                    st.divider()

# 8. Chat Logic executing directly against GeminiRAG instance
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not rag_system:
            st.error("System pipeline engines are currently uninitialized or configured incorrectly.")
        else:
            with st.status("Consulting the knowledge base...", expanded=True) as status:
                try:
                    # Call functions identically to how FastAPI processed them
                    relevant_docs = rag_system.retrieve_and_rerank(prompt)
                    answer = rag_system.generate(prompt, relevant_docs)
                    sources = [doc.page_content for doc in relevant_docs]
                    
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("🔍 View Evidence (Retrieved Context)"):
                            for idx, src in enumerate(sources):
                                st.caption(f"Source Chunk {idx+1}:")
                                st.write(src)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    status.update(label="RAG Engine Error", state="error")
                    st.error(f"Internal Pipeline Processing Failure: {e}")