#!/bin/sh

echo "What is your Docker Hub account username?"
read -r USERNAME
echo "What is your Docker Hub account password?"
stty -echo
read -r PASSWORD
stty echo

#USERNAME="username"
#PASSWORD="password"

APP_NAME="answers-to-questions"
TAG=$(git rev-parse --short HEAD)
APP_KEY="$USERNAME/$APP_NAME:$TAG"

docker login -u $USERNAME -p $PASSWORD 
docker compose build
docker tag $APP_NAME $APP_KEY
docker image rm $APP_NAME
docker push $APP_KEY
docker run -p 8501:8501 $APP_KEY

echo "Application available on DockerHub: https://hub.docker.com/repository/docker/$USERNAME/$APP_NAME/"
echo "Application started at http://localhost:8501..."
