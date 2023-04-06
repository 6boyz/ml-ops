# Папка второго задания по MLops 2023 spring
## Скрипт пайплайна

```
pipeline {
    agent any
    stages {
        stage('Data download') {
            steps {
                git credentialsId: 'github', url: 'https://github.com/6boyz/ml-ops'
                dir ('data_default') {
                    sh 'curl -LJO https://raw.githubusercontent.com/6boyz/ml-ops/master/static-files/house-pricing.csv'   
                }
            }
        }
        stage('Data creation') {
            steps { 
                sh 'python3 data_creation.py'
            }
        }
        stage('Model preprocessing') {
            steps {
                sh 'python3 model_preprocessing.py'
            }
        }
        stage('Model preparation') {
            steps {
                sh 'python3 model_preparation.py'
            }
        }
        stage('Model testing') {
            steps {
                sh 'python3 model_testing.py'
            }
        }
    }
}
```

*Всё что ниже - не работает*

## Развертывание пайплайна
Для работы пайплайна необходимо установить несколько дополнительных плагинов на Jenkins:
* ShiningPanda
* Pyenv Pipeline

### 1. Создание виртуального окружения

```
mkdir venvs
cd venvs
python3 -m venv ml-ops-base
cd ml-ops-base
curl -LJO https://raw.githubusercontent.com/6boyz/ml-ops/11029749686c4dd1eb2a9f5435ee3bf3bfb328c6/requirements.txt
bin/pip3 install -r requirements.txt
cd ../..
sudo chmod -R 777 venvs/.
```
### 2. Добавление виртуального окружения в Jenkins
Путь к венву: 
```
/home/USERNAME/venvs/ml-ops-base/bin/python
```
Название венва: **ml-ops-base**
### 3 Создание пайплайна в Jenkins
Скрипт пайплайна (креденшалсы не нужны):
```
pipeline {
    agent any
    stages {
        stage('Data download') {
            steps {
                git credentialsId: 'github', url: 'https://github.com/6boyz/ml-ops'
                dir ('data_default') {
                    sh 'curl -LJO https://raw.githubusercontent.com/6boyz/ml-ops/master/static-files/house-pricing.csv'   
                }
            }
        }
        stage('Data creation') {
            steps { 
                sh 'python3 data_creation.py'
            }
        }
        stage('Model preprocessing') {
            steps {
                sh 'python3 model_preprocessing.py'
            }
        }
        stage('Model preparation') {
            steps {
                sh 'python3 model_preparation.py'
            }
        }
        stage('Model testing') {
            steps {
                sh 'python3 model_testing.py'
            }
        }
    }
}
}
```