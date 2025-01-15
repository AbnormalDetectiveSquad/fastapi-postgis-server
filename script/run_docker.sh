#!/bin/bash
IMAGE_NAME=db-scheduler
TAG=$(date +%m%d)

echo "Starting Docker container..."
docker run -d \
    -p 8000:8000 \
    --env-file ../.env \
    --name ${IMAGE_NAME}-container \
    ${IMAGE_NAME}:${TAG}
