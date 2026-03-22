FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (saves ~2.7GB vs default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]
