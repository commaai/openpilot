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
whoami
sudo apt update
sudo apt install -y git'''
      }
    }

    stage('Build') {
      steps {
        sh '''
ls
git rev-parse --abbrev-ref HEAD
uname -a'''
      }
    }

  }
}