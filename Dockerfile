# Use Python 3.10 slim image as base (smaller size)
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install PyTorch CPU version first (much smaller than GPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py .

# Create directory for checkpoints
RUN mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: run training with CPU
CMD ["python", "train.py", "--accelerator", "cpu", "--epochs", "3"]