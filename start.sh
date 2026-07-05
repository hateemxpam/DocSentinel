#!/bin/bash

# Start FastAPI backend in the background on port 8000
# Do NOT redirect to a log file so we can see any errors/tracebacks in the Hugging Face Space logs tab
echo "Starting FastAPI backend on port 8000..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# Wait for FastAPI to start
echo "Waiting for backend to spin up..."
sleep 10

# Start Streamlit frontend on port 7860 (Hugging Face default port)
echo "Starting Streamlit frontend..."
streamlit run ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false
