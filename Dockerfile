FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY flashserve/ flashserve/
COPY web/ web/
COPY demo.py .
COPY .env.example .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Launch the server
CMD ["python3", "-m", "uvicorn", "flashserve.server:app", "--host", "0.0.0.0", "--port", "8000"]
