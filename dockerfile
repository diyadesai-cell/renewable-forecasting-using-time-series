FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (optional, but often needed for pandas, numpy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "web2.py", "--server.port=8501", "--server.address=0.0.0.0"]
