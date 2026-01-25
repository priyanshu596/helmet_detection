FROM python:3.10-slim

# --------------------------
# System dependencies
# --------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Environment
# --------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENCV_LOG_LEVEL=ERROR
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer|max_delay;0|buffer_size;102400"

WORKDIR /app

# --------------------------
# Python deps
# --------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
# App code
# --------------------------
COPY main.py .
COPY bytetrack.yaml .

CMD ["python", "main.py"]
