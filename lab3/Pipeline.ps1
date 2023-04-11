$ErrorActionPreference = "Stop" # Exit after first error code

$USERNAME = Read-Host 'What is your Docker Hub account username?'
$PASSWORD = Read-Host 'What is your Docker Hub account password?' -AsSecureString
$PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($PASSWORD))
$APP_NAME = "answers-to-questions"

#$USERNAME="username"
#$PASSWORD="password"

docker login -u $USERNAME -p $PASSWORD 

if (Test-Path "app") {
  Remove-Item "app" -Recurse -Force -Confirm:$false
}
git clone https://github.com/6boyz/project-practicum.git app
cd app
$TAG = git rev-parse --short HEAD
$APP_KEY = "$($USERNAME)/$($APP_NAME):$($TAG)"
cd ..

docker compose build
docker tag $APP_NAME $APP_KEY
docker image rm $APP_NAME
docker push $APP_KEY
docker run -d -p 8501:8501 $APP_KEY

Write-Output "Application: $APP_KEY"
Write-Output "On DockerHub: https://hub.docker.com/repository/docker/$($USERNAME)/$($APP_NAME)/"
Write-Output "Started at: http://localhost:8501"