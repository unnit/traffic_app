FROM python:3.12-slim

# 1. Install bare minimum OS dependencies for OpenCV Headless and Glis
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. Force install CPU-only PyTorch first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://pytorch.org

# 2. Install your standard requirements.txt list
RUN pip install --no-cache-dir -r requirements.txt

# 3. THE FIX: Forcibly overwrite any GUI OpenCV version with the headless server version
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY . .

EXPOSE 5002

CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]

