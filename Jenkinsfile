pipeline {
  agent {
    docker {
      image 'ubuntu:16.04'
    }

  }
  stages {
    stage('Install dependencies') {
      steps {
        sh '''
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
