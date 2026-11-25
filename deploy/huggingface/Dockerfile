# Hugging Face Spaces (Docker SDK) image.
# Builds the app straight from the public GitHub repo, so the Space only needs this file + README.md.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user                       # Spaces run containers as uid 1000

ARG REPO=https://github.com/mkalasaivan/ml_pipeline_event_mlops.git
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO} /app && chown -R user:user /app

WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

USER user
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 \
    MODELS_DIR=/app/models DATA_DIR=/app/data INFER_CONCURRENCY=2
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
