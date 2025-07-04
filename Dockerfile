# Gunakan Python base image
FROM python:3.10-slim

# Set direktori kerja
WORKDIR /app

# Copy file ke image
COPY . .

# Install ffmpeg dan dependensi sistem
RUN apt-get update && \
    apt-get install -y nano ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Jalankan API saat container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
