# Use Python 3.10 slim on amd64
FROM --platform=linux/amd64 python:3.10-slim

# Avoid Python buffering & ensure logs flush
ENV PYTHONUNBUFFERED=1

# Install OS deps (if needed by PyMuPDF or LightGBM)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (text + pickle + feature names)
COPY models/ ./models/

# Copy your source code and run script
COPY src/ ./src/
COPY run.sh .

# Set entrypoint
ENTRYPOINT ["./run.sh"]
