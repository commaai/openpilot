pipeline {
  agent {
    docker {
      image 'ubuntu:16.04'
      args '--user=root'
    }

  }
  stages {
    stage('Install dependencies') {
      steps {
        sh '''
apt update
apt install -y python python-pip
pip install paramiko
'''
      }
    }
    stage('EON Build/Test') {
      steps {
        lock(resource: "", label: 'eon', inversePrecedence: true, variable: 'eon_name', quantity: 1){
          timeout(time: 90, unit: 'MINUTES') {
            dir(path: 'selfdrive/test') {
              ansiColor('xterm') {
                sh 'ls #./release_build.py'
              }
            }
          }
        }
      }
    }

  }
}
