# Hugging Face Docker Spaces build/run on linux/amd64. Pinning the platform also
# lets Apple Silicon Macs resolve the same CPU-only PyTorch wheels locally.
FROM --platform=linux/amd64 python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PYTHONPATH=/app:/app/api \
    CHROMA_DB_PATH=/tmp/chroma \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers \
    NLTK_DATA=/home/user/nltk_data

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user:user requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Pin the spaCy model instead of using `spacy download`, which can fail during
# Docker builds when the compatibility lookup or redirect fetch is unavailable.
RUN python -m pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

RUN mkdir -p /home/user/nltk_data \
    && python -c "import nltk; nltk.download('punkt', download_dir='/home/user/nltk_data', quiet=True); nltk.download('punkt_tab', download_dir='/home/user/nltk_data', quiet=True)"

COPY --chown=user:user . /app

RUN mkdir -p /tmp/chroma /home/user/.cache/huggingface /home/user/nltk_data \
    && chown -R user:user /app /tmp/chroma /home/user

USER user

EXPOSE 7860

CMD ["uvicorn", "api.space_app:app", "--host", "0.0.0.0", "--port", "7860"]
