import streamlit as st
import requests
import json
import time
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Transformer RAG Expert",
    page_icon="🔬",
    layout="wide"
)

# 2. Custom CSS for a Professional Look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .st-emotion-cache-1c7n2ka { max-width: 95%; } /* Center content better */
    .sidebar-text { font-size: 14px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

METRICS_FILE_PATH = "/app/shared/metrics.json"

# 3. Configuration - Dynamically resolve container cross-talk routes
# Hugging Face internal networks communicate much more stably over 'localhost' or explicit '0.0.0.0' bindings.
#  Update to point directly to the shared local container gateway port
BACKEND_URL = "http://127.0.0.1:8002"
API_URL = BACKEND_URL

# 4. Sidebar - Dashboard & Management
with st.sidebar:
    st.title("📊 Control Center")
    
    # System Status Indicator
    try:
        health_check = requests.get(f"{API_URL}/", timeout=2)
        if health_check.status_code == 200:
            st.success("API Status: Online")
    except Exception:
        st.error("API Status: Offline (Check backend)")

    st.divider()
    
    # Metrics Section
    def load_metrics():
        try:
            with open(METRICS_FILE_PATH, "r") as f:
                data = json.load(f)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {"faithfulness": 0, "relevancy": 0, "last_run": "N/A"}

    metrics = load_metrics()

    st.subheader("Performance Metrics")
    col1, col2 = st.columns([1, 1])

    col1.metric("Faithfulness", f"{metrics.get('faithfulness', 0)*100:.0f}%")
    col2.metric("Relevancy", f"{metrics.get('relevancy', 0)*100:.0f}%")

    st.caption(f"Last benchmark run: {metrics.get('last_run', 'N/A')}")

    st.divider()
    
    st.subheader("🧪 Developer Tools")
    if st.button("🚀 Run System Benchmark", use_container_width=True):
        with st.status("Running RAGAS Evaluation...", expanded=False) as status:
            try:
                response = requests.post(f"{API_URL}/run-benchmark", timeout=5)
                
                if response.status_code == 200:
                    status.update(label="Benchmark Started!", state="complete")
                    st.toast("Evaluation triggered in background!", icon="✅")
                    time.sleep(2)
                    st.rerun()
                elif response.status_code == 422:
                    status.update(label="Validation Error", state="error")
                    st.error(f"Backend rejected request: {response.json()}")
                else:
                    status.update(label="Benchmark Failed", state="error")
                    st.error(f"Backend Error {response.status_code}: Check logs.")
            except Exception as e:
                status.update(label="Connection Error", state="error")
                st.error(f"Could not reach backend: {e}")

    st.divider()

    # Knowledge Management (The /upload Endpoint)
    st.subheader("📥 Knowledge Management")
    st.markdown('<p class="sidebar-text">Add new research papers to the Vector Store.</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
    
    if uploaded_file:
        if st.button("Process & Index Document", use_container_width=True):
            with st.spinner("Brain in progress... indexing..."):
                # Clean encapsulation of file binary buffers for multipart payload streams
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                    if response.status_code == 200:
                        st.balloons()
                        st.success(f"Indexed: {uploaded_file.name}")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error alternative')}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.divider()
    if st.button("Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. Main Chat UI Header
st.title("🔬 Research Assistant: Transformer Architecture")
st.info("Ask complex questions about Multi-Head Attention, Positional Encodings, or any uploaded papers.")

# 6. Initialize Chat History
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

# 7. Chat Input & Logic (The /ask Endpoint)
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Consulting the knowledge base...", expanded=True) as status:
            try:
                payload = {"prompt": prompt}
                response = requests.post(f"{API_URL}/ask", json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Evidence (Retrieved Context)"):
                        for idx, src in enumerate(sources):
                            st.caption(f"Source Chunk {idx+1}:")
                            st.write(src)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    status.update(label="API Error", state="error")
                    st.error("The backend returned an error. Please check your API logs.")
                    
            except Exception as e:
                status.update(label="Connection Failed", state="error")
                st.error(f"Could not connect to the API: {e}")