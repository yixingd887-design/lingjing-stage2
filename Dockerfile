FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 先安装 CPU 版 PyTorch，避免下载 CUDA 依赖
RUN pip install --upgrade pip && \
    pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .

RUN pip install -r requirements-docker.txt

COPY src/ ./src/
COPY web_app/ ./web_app/
COPY models/ ./models/

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "api:app", "--app-dir", "/app/web_app", "--host", "0.0.0.0", "--port", "8000"]
