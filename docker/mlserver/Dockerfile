FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    net-tools \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

ARG MLSERVER_VERSION=1.7.0
ARG MLSERVER_MLFLOW_VERSION=1.7.0
ARG MLFLOW_VERSION=2.22.0

RUN pip install --no-cache-dir \
    mlserver==${MLSERVER_VERSION} \
    mlserver-mlflow==${MLSERVER_MLFLOW_VERSION} \
    mlflow==${MLFLOW_VERSION} \
    xgboost \
    psutil \
    && rm -rf /root/.cache/pip

WORKDIR /tmp

CMD ["mlserver", "start", "/mnt/models/toy_xgboost_model"]
