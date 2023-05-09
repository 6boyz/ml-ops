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

