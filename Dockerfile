FROM python:3.12-slim

# Set environment variables to prevent TensorFlow import and set HF cache
ENV TRANSFORMERS_NO_TF=1
ENV USE_TF=0
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf_cache

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Switch to non-root user
USER user

# Expose Streamlit port
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.fileWatcherType=none"]
