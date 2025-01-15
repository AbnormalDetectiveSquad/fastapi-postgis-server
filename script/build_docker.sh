#!/bin/bash
IMAGE_NAME=db-scheduler
TAG=$(date +%m%d)

echo "build docker image $IMAGE_NAME:$TAG.."
docker build -t $IMAGE_NAME:$TAG ..
