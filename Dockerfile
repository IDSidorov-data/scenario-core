# Dockerfile — lightweight image for running Streamlit app (Python 3.11)
FROM python:3.11-slim

# System deps (add more if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt requirements-dev.txt* /app/

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# If you want dev deps inside the image for debugging uncomment next line
# RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy project
COPY . /app

# Expose default streamlit port
EXPOSE 8501

# Use $PORT provided by platforms (Heroku/Cloud) or default to 8501
ENV PORT=8501

# Recommended command — use sh -c to allow env var expansion
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]