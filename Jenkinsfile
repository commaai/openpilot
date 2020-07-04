pipeline {
  agent {
    docker {
      image 'python:3.7.3'
      args '--user=root'
    }
  }
  environment {
    COMMA_JWT = credentials('athena-test-jwt')
    CI_PUSH = "${env.BRANCH_NAME == 'master' ? 'master-ci' : ''}"
  }

  stages {
    stage('On-device Tests') {
      parallel {

        stage('Build') {
          steps {
            lock(resource: "", label: 'eon', inversePrecedence: true, variable: 'eon_ip', quantity: 1){
              timeout(time: 30, unit: 'MINUTES') {
                dir(path: 'selfdrive/test') {
                  sh 'pip install paramiko'
                  sh 'python phone_ci.py "cd release && ./build_devel.sh"'
                }
              }
            }
          }
        }

        stage('Replay Tests') {
          steps {
            lock(resource: "", label: 'eon2', inversePrecedence: true, variable: 'eon_ip', quantity: 1){
              timeout(time: 60, unit: 'MINUTES') {
                dir(path: 'selfdrive/test') {
                  sh 'pip install paramiko'
                  sh 'python phone_ci.py "cd selfdrive/test/process_replay && ./camera_replay.py"'
                }
              }
            }
          }
        }

      }
    }
  }
}
