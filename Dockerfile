FROM ghcr.io/tradingstrategy-ai/trade-executor:latest
USER root
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN python -m pip install --no-cache-dir fastapi uvicorn zelos-demeter
COPY app.py /app/app.py
ENTRYPOINT []
CMD ["sh","-c","uvicorn app:api --host 0.0.0.0 --port ${PORT:-8000}"]
