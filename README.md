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

# 📊 Document Analyzer Project (RAG)

A comprehensive, end-to-end Machine Learning application designed to perform sentiment analysis. This repository includes model training, evaluation logs with MLflow, and a containerized interactive Streamlit dashboard deployed to Hugging Face Spaces.

---

## 🚀 Live Demo
Check out the interactive dashboard live on Hugging Face Spaces: 
👉 **https://huggingface.co/spaces/Aswathi007/Document-Analyzer-RAG**

---

## 📁 Repository Structure

```text
SENTIMENT ANALYSIS/
├── app/                  # Streamlit dashboard application files
│   └── app.py            # Main entry point for the web app
├── data/                 # Raw and processed datasets
├── models/               # Model artifacts (ONNX, checkpoints)
│   ├── onnx_model/
│   └── sentiment_model/
├── notebooks/            # Jupyter notebooks for EDA and prototyping
├── src/                  # Core source code (preprocessing, modeling)
├── tests/                # Unit and integration tests
├── .dockerignore         # Docker build exclusion rules
├── .gitignore            # Git version control exclusion rules
├── Dockerfile            # Blueprint for the containerized environment
├── requirements.txt      # Python dependencies
└── train_model.py        # Core script for pipeline model training

```
---

## 🛠️ Tech Stack

Language: Python 3.10
Frontend/UI: Streamlit
Deployment & Containerization: Docker, Hugging Face Spaces
Experiment Tracking: MLflow
Version Control: GitHub

---

## ⚙️ Local Setup and Installation
If you want to run this project locally without Docker, follow these steps:

1. Clone the repository
Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd "SENTIMENT ANALYSIS"
2. Set up a virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Configure Environment Variables
Create a .env file in the root directory and add any necessary credentials (e.g., your Google GenAI keys):
```
GEMINI_API_KEY=your_api_key_here
```
6. Run the Application

```
streamlit run app/app.py
```
---

## 🐳 Running with Docker
This project is completely containerized. To build and run the application locally inside its isolated environment:

Build the Docker Image
```
docker build -t sentiment-streamlit-app:latest .
```
Run the Docker Container
```
docker run -p 8501:8501 --env-file .env sentiment-streamlit-app:latest
```
Once started, navigate to http://localhost:8501 in your web browser.

---

## 🧪 Testing
To run the test suites locally, ensure you are in your virtual environment and run:

```
pytest
```
---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

### What to do next:
1. Open your local `README.md` file.
2. Replace everything inside it with the text above.
3. Replace placeholders like `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual links.
4. Save and run your Git push commands to update both GitHub and Hugging Face:

```bash
git add README.md
git commit -m "docs: add comprehensive README with HF metadata"
git push origin main
git push hf main
