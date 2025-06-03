# ---- Base image ----
    FROM python:3.10-slim

    # ---- Set environment variables ----
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    
    # ---- Create app directory ----
    WORKDIR /app
    
    # ---- Copy app files ----
    COPY . /app
    
    # ---- Install system dependencies ----
    RUN apt-get update && apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    
    # ---- Install Python dependencies ----
    RUN pip install --upgrade pip
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---- Set default command ----
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
    