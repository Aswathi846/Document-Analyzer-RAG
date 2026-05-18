#!/bin/bash

# 1. Start ChromaDB in the background
echo "Starting ChromaDB..."
chroma run --host 127.0.0.1 --port 8000 --path ./chroma_data &

# 2. Wait for DB to wake up cleanly
sleep 4

# 3. Start the FastAPI backend in the background
echo "Starting FastAPI Backend..."
python backend/main_final.py &

# 4. Wait for Backend port allocation
sleep 3

# 5. Start Streamlit on Hugging Face's mandatory port (7860)
echo "Starting Streamlit UI on Hugging Face port..."
streamlit run frontend/app_final.py --server.port=7860 --server.address=0.0.0.0