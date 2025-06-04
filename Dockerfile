# ---- Base image ----
FROM python:3.10-slim

# ---- Set environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Create app directory ----
WORKDIR /app

# ---- Copy app files ----
COPY . /app

# ---- Install Python dependencies ----
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---- Start the FastAPI app with Uvicorn ----
CMD ["python", "-m", "uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
