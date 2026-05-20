FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend modules (.dockerignore excludes __pycache__, .venv, etc.):
#   main.py, model.py, research.py, refusal_pairs.py, over_refusal_pairs.py
#   refusal_bench/ (the bench harness + 6 techniques + harmfulness probe)
#   scripts/ (the data build scripts; not invoked at runtime but committed)
COPY . .

RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]
