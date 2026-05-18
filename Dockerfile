FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling Chroma/pydantic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install everything directly into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the clean application directories
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY src/ ./src/
# Create the directory structure explicitly inside the container 
RUN mkdir -p data/processed data/processed_data data/raw_data data/uploads /app/shared

# Create the specific files with basic seed contents so the app doesn't crash on boot
RUN echo "Initial evaluation data" > data/processed/evaluation_report.csv
RUN echo "Cleaned text documents go here" > data/processed_data/docs_clean.txt
RUN echo "Raw documents go here" > data/raw_data/docs.txt

COPY start.sh .

# Give execute permissions to our startup manager script
RUN chmod +x start.sh

# Hugging Face Spaces strictly routes external traffic through port 7860
EXPOSE 7860

CMD ["./start.sh"]