# Gunakan Python base image
FROM python:3.10-slim

# Set direktori kerja
WORKDIR /app

# Install system dependencies terlebih dahulu
RUN apt-get update && \
    apt-get install -y \
    nano \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt terlebih dahulu untuk better caching
COPY requirements.txt .

# Install dependensi Python dengan optimasi
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi
COPY . .

EXPOSE 5000

# Jalankan API saat container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
