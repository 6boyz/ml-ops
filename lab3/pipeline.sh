#!/usr/bin/env bash
set -Eeuo pipefail # Exit after first error code

echo "What is your Docker Hub account username?"
read -r USERNAME
echo "What is your Docker Hub account password?"
stty -echo
read -r PASSWORD
stty echo

#USERNAME="username"
#PASSWORD="password"
APP_NAME="answers-to-questions"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
APP_PATH="$SCRIPT_DIR/app/"

docker login -u $USERNAME -p $PASSWORD 

if [ -d "$APP_PATH" ]; then rm -Rf $APP_PATH; fi
git clone https://github.com/6boyz/project-practicum.git app
cd $SCRIPT_DIR
TAG=$(git rev-parse --short HEAD)
APP_KEY="$USERNAME/$APP_NAME:$TAG"
cd -

docker compose build
docker tag $APP_NAME $APP_KEY
docker image rm $APP_NAME
docker push $APP_KEY
docker run -d -p 8501:8501 $APP_KEY

echo "Application available on DockerHub: https://hub.docker.com/repository/docker/$USERNAME/$APP_NAME/"
echo "Application started at http://localhost:8501..."