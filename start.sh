#!/bin/bash

# Start the FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8080 &

# Start the Streamlit frontend
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
