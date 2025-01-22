FROM python:3.12-slim

WORKDIR /

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 실행 관련 필수 패키지 설치
RUN pip install --no-cache-dir pytz && \
    pip install --no-cache-dir python-dateutil && \
    pip install --no-cache-dir scipy && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir scikit-learn && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY ./app /app

WORKDIR /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]