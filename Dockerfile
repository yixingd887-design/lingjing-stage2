
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements-docker.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

# 安装 Linux CPU 版 PyTorch，供 YOLO 使用
RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY web_app/ ./web_app/
COPY models/ ./models/

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "api:app", "--app-dir", "/app/web_app", "--host", "0.0.0.0", "--port", "8000"]
