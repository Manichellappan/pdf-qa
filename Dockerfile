# Dockerfile for Streamlit + LangChain + FAISS app (CPU)
FROM python:3.10-slim

# set workdir
WORKDIR /app

# install system deps needed for some Python packages (faiss, torch wheels, opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy app files
COPY . .

# upgrade pip and install Python deps
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# expose Streamlit default port used by HF spaces
EXPOSE 7860

# run Streamlit app (adjust filename if your entrypoint is different)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
