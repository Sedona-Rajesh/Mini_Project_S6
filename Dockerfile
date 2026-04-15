FROM python:3.11-slim

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
COPY outputs/ ./outputs/

# Start the Streamlit application (FIXED)
CMD ["sh", "-c", "streamlit run eeg_dss/app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0"]