# ---- Base image ----
FROM python:3.10-slim

# ---- Install system-level dependencies (OpenCV fix) ----
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ---- Set environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Create app directory ----
WORKDIR /app

# ---- Copy project files ----
COPY . /app

# ---- Install Python dependencies ----
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Optional but good to expose for docs
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
