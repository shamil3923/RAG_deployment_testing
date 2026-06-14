FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# copy and install requirements first to leverage layer caching
COPY backend/requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# copy application code
COPY backend/ /app/

# persist Chroma DB between restarts
VOLUME ["/app/chroma_db"]

EXPOSE 8000

# Run Uvicorn; in production consider using gunicorn+uvicorn workers or a process manager
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
