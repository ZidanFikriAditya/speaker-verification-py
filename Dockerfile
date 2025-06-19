# Gunakan Python base image
FROM python:3.10-slim

# Install ffmpeg dan dependensi sistem
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# Copy file ke image
COPY requirements.txt .
COPY .env .
COPY main.py .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan API saat container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
