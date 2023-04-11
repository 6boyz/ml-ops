$USERNAME = Read-Host 'What is your Docker Hub account username?'
$PASSWORD = Read-Host 'What is your Docker Hub account password?' -AsSecureString

$PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($PASSWORD))

#$USERNAME="username"
#$PASSWORD="password"

$APP_NAME = "answers-to-questions"

git clone https://github.com/6boyz/project-practicum.git app
cd app

$TAG = git rev-parse --short HEAD
$APP_KEY = "$($USERNAME)/$($APP_NAME):$($TAG)"

docker login -u $USERNAME -p $PASSWORD 
docker compose build
docker tag $APP_NAME $APP_KEY
docker image rm $APP_NAME
docker push $APP_KEY
docker run -p 8501:8501 $APP_KEY

Write-Output "Application available on DockerHub: https://hub.docker.com/repository/docker/$($USERNAME)/$($APP_NAME)/"
Write-Output 'Application started at http://localhost:8501...'
