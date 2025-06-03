# ---- Base image ----
FROM python:3.10-slim

# ---- Set environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Set working directory ----
WORKDIR /app

# ---- Install system dependencies ----
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy project files ----
COPY . /app

# ---- Upgrade pip and install Python dependencies ----
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---- Run Uvicorn server on port 8080 ----
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
