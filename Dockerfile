# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libsqlite3-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data offline
RUN python -m nltk.downloader punkt stopwords

# Set environment variables (optional override at runtime)
ENV INPUT_JSON=/app/challenge1b_input.json
ENV PDF_FOLDER=/app/pdf
ENV OUTPUT_PATH=/app/result.json

# Run script
CMD ["python", "chall2.py"]
