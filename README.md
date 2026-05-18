---
title: Document Analyzer RAG
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
python_version: 3.11
app_file: frontend/app_final.py  
pinned: false
---

# 🔬 Advanced RAG Research Assistant: Transformer Architecture

A comprehensive, end-to-end Machine Learning application designed to perform sentiment analysis. This repository includes model training, evaluation logs with MLflow, and a containerized interactive Streamlit dashboard deployed to Hugging Face Spaces.

---
## 🚀 Live Demo

• Check out the interactive dashboard live on Hugging Face Spaces: 👉 **https://huggingface.co/spaces/Aswathi007/Document-Analyzer-RAG**

• Langsmith Observability Dashboard: 👉 **https://smith.langchain.com/o/5c81bcd3-13e7-4343-a02b-4b2607ae6f8e/projects/p/3621b448-8a85-40b8-b0bf-218a91aa76e8**

---
## 📸 System Interface
Below is a live look at the production interface tracking and rendering advanced context-retrieval spans:

<img width="1825" height="796" alt="image" src="https://github.com/user-attachments/assets/0d166c60-5e48-4054-8db2-25970c3afcfc" />

---

## 🏗️ Production Architecture & Engineering Evolution
The application has transitioned from a loose local script collection into a hardened, production-ready Monolith-in-a-Box architecture optimized for free-tier cloud constraints (like Hugging Face Spaces).

**1. Architectural Re-engineering (Separation of Concerns)**
Testing isolation: Evaluation utilities were refactored completely out of the local /tests layout and moved into src/evaluation_utils.py. This ensures production container images remain clean, lightweight, and completely decoupled from testing dependencies, preventing ModuleNotFoundError during cloud compilation.
Stateful Volume Binding: Configured specific Docker storage mappings for directory persistence (chroma_db/, data/, and metrics.json), allowing seamless hot-reloading across the front-to-back architecture.

**2. Multi-Process Supervisor Orchestration**
To deploy on free cloud architectures that restrict environments to a single open ingress port (7860), the system utilizes an optimized Linux Supervisor multi-process manager. This lightweight orchestration layer boots and manages both the background FastAPI application server and the interactive Streamlit user viewport concurrently inside a unified runtime layer.

**3. Deep LLMOps Observability (LangSmith Integration)**
The system moves past disjointed logging by executing parent context nesting. Using the native @traceable decorator framework linked directly to custom environment hooks, individual spans (RAG_Retriever and RAG_Generator) are collected, grouped, and streamed out as a singular synchronized tree trace execution graph.

---
## 📁 Repository Structure

```text
project-root/
├── .github/
│   └── workflows/
│       └── main.yml          # Automated CI/CD Git Sync Pipeline
├── src/
│   ├── data_processor.py       # Raw to Processed ETL logic
│   ├── ingestion.py            # Chunking, vector indexing & Embedding generation
│   ├── pipeline.py             # RAG Orchestration (Nesting Retriever + Generator)
│   └── evaluation.py     # Relocated RAGAS Benchmark Evaluators
├── backend/
│   └── main_final.py                 # FastAPI REST Engine & Background Worker tasks
│   └── Dockerfile
├── frontend/
│   └── app_final.py                  # Streamlit Viewport Dashboard UX
│   └── Dockerfile.frontend
├── data/
│   ├── raw/                    # Source technical manuscripts
│   └── uploads/                # Dynamic user ingestion folder
├── chroma_db/                  # Persistent Vector database snapshot local disk
├── Dockerfile                  # Production Monolith Supervisor Image blueprint
├── docker-compose.yml          # Local container development coordinator
├── metrics.json                # Live-updating system performance benchmark storage
└── requirements.txt            # Explicitly pinned application dependencies

```
---

## 🛠️ Tech Stack

• Language: Python 3.10

• Frontend/UI: Streamlit

• Deployment & Containerization: Docker, Hugging Face Spaces

• Experiment Tracking: MLflow

• Version Control: GitHub

---

## 📈 Engineering Rigor & Features

• **Two-Stage Retrieval (Recall vs Precision)**: Fetches the top 10 most candidate document nodes via vector similarity (Stage 1), then passes them through a localized Cross-Encoder model to bubble up the most contextually relevant information (Stage 2). This eliminates "lost-in-the-middle" LLM context degradation.

• **Non-Blocking Background Metrics**: The /run-benchmark pipeline offloads compute-heavy evaluations into an asynchronous BackgroundTasks queue. This permits immediate frontend server response while deep model evaluations continue running under the hood.

• **Secure Environment Architecture**: Zero-hardcoded credentials. Sensitive access variables are routed strictly out of ephemeral container memory configurations, keeping internal cloud keys protected.

---

## 🔄 Automated CI/CD Deployment

This repository uses **GitHub Actions** for continuous integration and continuous deployment (CI/CD). You don't need to manually push your code to both GitHub and Hugging Face. Whenever code is merged into the `main` branch, a workflow automatically builds, synchronizes, and redeploys the app to Hugging Face Spaces.

### How to set up the CI/CD Pipeline:

1. **Get your Hugging Face Token:**
   * Go to your Hugging Face Account Settings -> **Access Tokens**.
   * Create a new token with `Write` access and copy it.

2. **Add Secrets to GitHub:**
   * Go to your repository on GitHub.
   * Navigate to **Settings** -> **Secrets and variables** -> **Actions**.
   * Click **New repository secret** and add:
     * `HF_TOKEN`: *Paste your Hugging Face Write Token*

3. **Workflow Configuration:**
   The workflow file is located at `.github/workflows/deploy.yml` and runs automatically on every push to `main`:

```yaml
name: Sync to Hugging Face Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Mirror to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config --global user.email "actions@github.com"
          git config --global user.name "GitHub Actions"
          
          # Use an authenticated header injection instead of plain text URLs
          git remote add hf https://x-token-auth:${HF_TOKEN}@huggingface.co/spaces/Aswathi007/Document-Analyzer-RAG.git || \
          git remote set-url hf https://x-token-auth:${HF_TOKEN}@huggingface.co/spaces/Aswathi007/Document-Analyzer-RAG.git
          
          # Force sync changes smoothly across remotes
          git push --force hf main
```
---

## 📄 License
This project is licensed under the MIT License.

---
