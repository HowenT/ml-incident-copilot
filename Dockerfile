FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend backend
COPY frontend frontend
COPY .streamlit .streamlit

EXPOSE 8000 8501
# The API seeds the demo environment on first start (~20 s).
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
