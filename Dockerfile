FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-min.txt ./
RUN pip install --no-cache-dir -r requirements-min.txt

COPY . .

CMD ["python", "scripts/run_assistant.py", "--text", "Set temperature to 72"]
