FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY bome_backtest.py /app/bome_backtest.py
CMD ["python","/app/bome_backtest.py"]
