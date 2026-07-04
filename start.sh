#!/bin/bash

# Start FastAPI backend in the background on port 8000
echo "Starting FastAPI backend..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 > fastapi.log 2>&1 &

# Wait for FastAPI to start
echo "Waiting for backend to spin up..."
sleep 5

# Start Streamlit frontend on port 7860 (Hugging Face default port)
echo "Starting Streamlit frontend..."
streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0
