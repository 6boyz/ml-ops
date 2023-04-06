USERNAME="username"
PASSWORD="password"

APP_NAME="answers-to-questions"
TAG=$(git rev-parse --short HEAD)
APP_KEY="$($USERNAME)/$($APP_NAME):$($TAG)"

docker login -u $USERNAME -p $PASSWORD 
docker compose build
docker tag $APP_NAME $APP_KEY
docker push $APP_KEY