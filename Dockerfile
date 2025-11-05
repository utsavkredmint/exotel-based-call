# ==========================
#   Stage 1 — Base Image
# ==========================
FROM python:3.13-slim AS base

# Avoid Python buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install OS dependencies for audio + build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for caching
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the project
COPY . .

# Expose Streamlit default port
EXPOSE 3000

# ==========================
#   Stage 2 — Production Run
# ==========================
CMD ["streamlit", "run", "app.py", "--server.port=3000", "--server.address=0.0.0.0"]
