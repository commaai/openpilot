pipeline {
  agent {
    docker {
      image 'ubuntu:16.04'
      args '--user=root'
    }

  }
  stages {
    stage('EON Build/Test') {
      steps {
        lock(resource: "", label: 'eon', inversePrecedence: true, variable: 'eon_name', quantity: 1){
          timeout(time: 30, unit: 'MINUTES') {
            dir(path: 'selfdrive/test') {
              sh 'apt update'
              sh 'apt install -y python python-pip'
              sh 'pip install paramiko'
              sh 'python release/remote_build.py'
            }
          }
        }
      }
    }
  }
}
