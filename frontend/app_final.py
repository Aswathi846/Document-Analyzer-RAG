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



# Use localhost instead of the service name 'backend'-for local usage

#API_URL = "http://127.0.0.1:8002"



METRICS_FILE_PATH = "/app/shared/metrics.json"

# 3. Configuration - Explicitly map to your internal FastAPI container port
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002")
API_URL = BACKEND_URL



# 4. Sidebar - Dashboard & Management

with st.sidebar:

    st.title("📊 Control Center")

   

    # System Status Indicator

    try:

        health_check = requests.get(f"{API_URL}/")

        if health_check.status_code == 200:

            st.success("API Status: Online")

    except:

        st.error("API Status: Offline (Check backend)")



    st.divider()

   

    # Metrics Section

    # Metrics Section
    # Metrics Section
    def load_metrics():
        try:
            # Use the absolute container volume path variable
            with open(METRICS_FILE_PATH, "r") as f:
                data = json.load(f)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback if benchmark hasn't run yet or file is empty
            return {"faithfulness": 0, "relevancy": 0, "last_run": "N/A"}



    metrics = load_metrics()



    # In your sidebar code:

    st.subheader("Performance Metrics")

    col1, col2 = st.columns([1, 1])

    #col1, col2 = st.columns(2)



    # Display data from your actual evaluation runs

    col1.metric("Faithfulness", f"{metrics['faithfulness']*100:.0f}%")

    col2.metric("Relevancy", f"{metrics['relevancy']*100:.0f}%")



    st.caption(f"Last benchmark run: {metrics['last_run']}")



   

    st.divider()

   

    #Adding a "Run Benchmark" button to the sidebar

    st.subheader("🧪 Developer Tools")

    if st.button("🚀 Run System Benchmark", use_container_width=True):

        with st.status("Running RAGAS Evaluation...", expanded=False) as status:

            try:

                response = requests.post(f"{API_URL}/run-benchmark")

               

                if response.status_code == 200:

                    status.update(label="Benchmark Started!", state="complete")

                    st.toast("Evaluation triggered in background!", icon="✅")

                    # Wait a moment so the user sees the success before the rerun

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

                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                try:

                    response = requests.post(f"{API_URL}/upload", files=files)

                    if response.status_code == 200:

                        st.balloons()

                        st.success(f"Indexed: {uploaded_file.name}")

                    else:

                        st.error(f"Error: {response.json().get('detail')}")

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

    # Add user message to history

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):

        st.markdown(prompt)



    # Call API for response

    with st.chat_message("assistant"):

        with st.status("Consulting the knowledge base...", expanded=True) as status:

            try:

                # API Call to /ask

                payload = {"prompt": prompt}

                response = requests.post(f"{API_URL}/ask", json=payload)

               

                if response.status_code == 200:

                    data = response.json()

                    answer = data["answer"]

                    sources = data["sources"]

                   

                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                   

                    # Display Answer

                    st.markdown(answer)

                   

                    # Display Sources

                    with st.expander("🔍 View Evidence (Retrieved Context)"):

                        for idx, src in enumerate(sources):

                            st.caption(f"Source Chunk {idx+1}:")

                            st.write(src)

                   

                    # Save to session state

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