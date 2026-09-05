FROM python:3.11-slim

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the FastEmbed ONNX models into image so the container starts immediately and offline
ENV FASTEMBED_CACHE_PATH=/root/.cache/fastembed
RUN python3 -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# Copy application source code and OpenAPI schema
COPY config.py brain.py server.py openapi.json ./

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=5006
ENV QDRANT_URL=http://qdrant:6333
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV ORT_INTRA_OP_NUM_THREADS=1
ENV ONNX_THREADS=1

EXPOSE 5006

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5006/health || exit 1

CMD ["python", "server.py"]
