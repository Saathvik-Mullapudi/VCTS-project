FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, GStreamer, PyGObject, and Cairo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3-dev \
    libgirepository1.0-dev \
    libglib2.0-dev \
    gobject-introspection \
    libcairo2-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstrtspserver-1.0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer caching: Copy requirements first and install Python packages
COPY requirements.txt .

# Upgrade pip and install Python dependencies (pin PyGObject<3.50 for Debian 12 GLib 1.0 compatibility)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "PyGObject<3.50.0" pycairo

# Copy application files into the image
COPY main.py .
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/ ./data/

# Create runtime output directories
RUN mkdir -p outputs/logs outputs/metrics outputs/videos

# Expose RTSP server port (8554) and UDP streaming port (5006)
EXPOSE 8554 5006/udp

# Default entry command
CMD ["python", "main.py", "--video", "data/videos/fixed.mp4", "--profile"]
