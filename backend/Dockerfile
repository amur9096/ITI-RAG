FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and data
COPY backend/ /app/backend/
COPY data/ /app/data/

ENV PYTHONPATH=/app
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV CHROMA_PATH=backend/data/vector_store

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
