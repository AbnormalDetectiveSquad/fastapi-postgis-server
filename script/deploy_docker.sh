#!/bin/bash
IMAGE_NAME=db-scheduler
TAG=$(date +%m%d)
SERVER_USER="ubuntu"
SERVER_IP="3.34.211.86"
SERVER_PATH="/home/ubuntu"

# 이미지 저장
echo "Saving docker image..."
docker save ${IMAGE_NAME}:${TAG} > ${IMAGE_NAME}_${TAG}.tar

# 서버로 전송
echo "Copying to server..."
scp ${IMAGE_NAME}_${TAG}.tar ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/

# 로컬의 tar 파일 삭제
rm ${IMAGE_NAME}_${TAG}.tar

# 서버에서 이미지 로드 및 실행
echo "Loading and running on server..."
ssh ${SERVER_USER}@${SERVER_IP} "
    docker load < ${SERVER_PATH}/${IMAGE_NAME}_${TAG}.tar && \
    rm ${SERVER_PATH}/${IMAGE_NAME}_${TAG}.tar && \
    docker run -d \
        -p 8000:8000 \
        --env-file .env \
        --name ${IMAGE_NAME}-container \
        ${IMAGE_NAME}:${TAG}
"

echo "Deployment completed!"