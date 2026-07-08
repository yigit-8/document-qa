FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Vector store is populated at runtime via /ingest
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
