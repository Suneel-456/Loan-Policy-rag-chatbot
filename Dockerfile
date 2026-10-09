# FastAPI + RAG service. Models are NOT baked into the image (they are ~3 GB);
# they download on first start into the /models/hf volume and are reused afterwards.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    VECTORSTORE_PATH=/app/vectorstore/index.json

WORKDIR /app

# CPU-only PyTorch first (much smaller than the default CUDA build)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt \
    && python -m spacy download en_core_web_sm

# Optional Langfuse tracing client: docker build --build-arg WITH_LANGFUSE=1 ...
ARG WITH_LANGFUSE=0
COPY requirements-observability.txt .
RUN if [ "$WITH_LANGFUSE" = "1" ]; then pip install -r requirements-observability.txt; fi

COPY src ./src
COPY data ./data

RUN useradd -m -u 1000 appuser \
    && mkdir -p /models/hf /app/vectorstore \
    && chown -R appuser:appuser /models /app
USER appuser

EXPOSE 8000

# First start downloads the models, so allow a long start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]