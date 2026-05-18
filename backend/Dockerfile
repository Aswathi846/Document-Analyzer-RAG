# Use a slim version for a smaller attack surface
FROM python:3.11-slim

# Install curl for the healthcheck tool
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 1. Create a non-privileged user
RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser

WORKDIR /app

# 2. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application code (context is root, so this copies your whole project folder)
COPY . .

# 4. Give the non-root user ownership of the app directory
RUN chown -R appuser:appuser /app

# 5. Switch to the non-root user
USER appuser

# Expose our designated backend port
EXPOSE 8002

# 6. Correct Healthcheck (Pings our FastAPI / status endpoint on port 8002)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/ || exit 1

# 7. Main execution command to launch the app
CMD ["python", "backend/main_final.py"]