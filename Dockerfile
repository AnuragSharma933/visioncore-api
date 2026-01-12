FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Create weights directory
RUN mkdir -p weights

# Expose port 7860 (required by HF Spaces)
EXPOSE 7860

# Run the app
CMD ["python", "main.py"]
