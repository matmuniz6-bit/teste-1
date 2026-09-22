FROM ghcr.io/tradingstrategy-ai/trade-executor:latest

USER root
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/usr/src/trade-executor:/usr/src/trade-executor/deps/web3-ethereum-defi:/usr/src/trade-executor/deps/trading-strategy"

RUN python -m pip install --no-cache-dir fastapi uvicorn zelos-demeter

COPY app.py native_strategy.py /app/

ENTRYPOINT []
CMD ["sh","-c","cd /usr/src/trade-executor && python -m uvicorn --app-dir /app app:api --host 0.0.0.0 --port ${PORT:-8000}"]
