# ---- Base image ----
FROM python:3.10-slim

# ---- Install system-level dependencies ----
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Set environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Create and move to app directory ----
WORKDIR /app

# ---- Copy files ----
COPY . /app

# ---- Install Python dependencies ----
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---- Expose port for Cloud Run ----
EXPOSE 8080

# ---- Start the FastAPI app with Uvicorn ----
CMD ["python", "main.py"]
