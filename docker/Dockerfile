FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source and install
COPY . .
RUN pip install --no-cache-dir .

EXPOSE 8100

CMD ["python", "-m", "src.cli", "serve", "--host", "0.0.0.0", "--port", "8100"]
