# Use official slim Python image
FROM python:3.11-slim

# Install system dependencies (build-essential for compiling some python deps if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Make start script executable and fix CRLF line endings (Windows -> Linux)
RUN chmod +x start.sh && sed -i 's/\r//' start.sh

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the launch script
CMD ["./start.sh"]
