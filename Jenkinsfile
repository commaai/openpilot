pipeline {
  agent {
    docker {
      image 'ubuntu:16.04'
    }

  }
  stages {
    stage('Build') {
      steps {
        sh '''ls
git rev-parse --abbrev-ref HEAD
uname -a'''
      }
    }

    stage('Install dependencies') {
      steps {
        sh '''apt update
apt install git'''
      }
    }

  }
}