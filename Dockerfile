# LabSentinel-RAG — container image
#   docker compose up --build        (recommended; see docker-compose.yml)
#   docker build -t labsentinel . && docker run --env-file .env -p 7860:7860 labsentinel
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_PORT=7860

# ffmpeg: video URL downloads (yt-dlp merge) · libgl/libglib: OpenCV runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default wheel pulls ~2.5 GB of CUDA libraries)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the container starts offline
ARG EMBED_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"

COPY . .

# Run as a non-root user; data/ and outputs/ are usually mounted as volumes
RUN useradd --create-home --uid 1000 app \
 && mkdir -p data/sops data/manuals data/incidents data/videos data/embeddings outputs \
 && chown -R app:app /app /opt/hf-cache
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"GRADIO_PORT\",\"7860\")}/', timeout=4)" || exit 1

CMD ["python", "app.py"]
