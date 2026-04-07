FROM python:3.10-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

CMD ["python", "app.py"]