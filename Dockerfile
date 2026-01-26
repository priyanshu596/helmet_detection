# ==========================
# Base image (CPU, small, stable)
# ==========================
FROM python:3.10-slim

# ==========================
# Environment
# ==========================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENCV_LOG_LEVEL=ERROR

# ==========================
# System dependencies
# ==========================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================
# Working directory
# ==========================
WORKDIR /app

# ==========================
# Python dependencies
# ==========================
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    ultralytics==8.* \
    opencv-python-headless \
    flask \
    numpy \
    torch --index-url https://download.pytorch.org/whl/cpu

# ==========================
# Copy project files
# ==========================
COPY . /app

# ==========================
# Create output dirs (safe)
# ==========================
RUN mkdir -p captures/helmet captures/no_helmet

# ==========================
# Expose Flask port
# ==========================
EXPOSE 5000

# ==========================
# Run application
# ==========================
CMD ["python", "main.py"]
