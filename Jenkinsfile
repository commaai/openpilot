pipeline {
  agent {
    docker {
      image 'python:3.7.3'
      args '--user=root'
    }
  }
  environment {
    COMMA_JWT = credentials('athena-test-jwt')
    PUSH = "${env.BRNACH_NAME == 'master' ? 'master-ci' : ''}"
  }

  if (!['master-ci', 'devel', 'release2', 'release2-staging', 'dashcam', 'dashcam-staging'].contains(env.BRANCH_NAME)) {
    stages {
      stage('On-device Tests') {

        parallel {

          stage('Build') {
            steps {
              lock(resource: "", label: 'eon', inversePrecedence: true, variable: 'eon_ip', quantity: 1){
                timeout(time: 30, unit: 'MINUTES') {
                  dir(path: 'selfdrive/test') {
                    sh 'pip install paramiko'

                    if (env.BRANCH_NAME != "devel-staging")  {
                      sh 'python phone_ci.py "cd release && ./build_devel.sh"'
                    } else {
                      // build release2-staging and dashcam-staging
                    }
                  }
                }
              }
            }
          }

          stage('Replay Tests') {
            steps {
              lock(resource: "", label: 'eon2', inversePrecedence: true, variable: 'eon_ip', quantity: 1){
                timeout(time: 45, unit: 'MINUTES') {
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
}
