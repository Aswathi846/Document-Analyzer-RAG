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
COPY data/ ./data/
COPY start.sh .

# Give execute permissions to our startup manager script
RUN chmod +x start.sh

# Hugging Face Spaces strictly routes external traffic through port 7860
EXPOSE 7860

CMD ["./start.sh"]