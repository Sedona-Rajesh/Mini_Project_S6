FROM python:3.10-slim

# Expose Streamlit default port
EXPOSE 8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging files and install reqs
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Copy application source code
COPY eeg_dss/ ./eeg_dss/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Copy the trained models (assuming they sit in outputs/)
# Be cautious if output/ is huge, but we need the artifacts for Streamlit
COPY outputs/ ./outputs/

# Add healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "eeg_dss/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
