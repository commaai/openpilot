pipeline {
  agent any
  environment {
    AUTHOR = """${sh(
                returnStdout: true,
                script: "git --no-pager show -s --format='%an' ${GIT_COMMIT}"
             ).trim()}"""
    GIT_COMMIT_DESC =  """${sh(
                          returnStdout: true,
                          script: "git log --format=%B -n 1 ${GIT_COMMIT}"
                       ).trim()}"""
    GIT_COMMIT_SHORT = """${sh(
                          returnStdout: true,
                          script: "git rev-parse --short=8 ${GIT_COMMIT}"
                       ).trim()}"""
    COMMA_JWT = credentials('athena-test-jwt')
  }
  stages {
    stage('Builds') {
      parallel {
        stage('EON Build/Test') {
          steps {
            lock(resource: "", label: 'eon', inversePrecedence: true, variable: 'eon_name', quantity: 1){
              timeout(time: 90, unit: 'MINUTES') {
                dir(path: 'selfdrive/test') {
                  ansiColor('xterm') {
                    sh './release_build.py'
                  }
                }
              }
            }
          }
        }
        stage('LEON Build/Test') {
          steps {
            lock(resource: "", label: 'leon', inversePrecedence: true, variable: 'leon_name', quantity: 1){
              timeout(time: 90, unit: 'MINUTES') {
                dir(path: 'selfdrive/test') {
                  ansiColor('xterm') {
                    sh './release_build.py'
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  post {
    failure {
      slackSend(color:'danger', message:"Failed: one » ${env.JOB_NAME} [${env.BUILD_NUMBER}] (<${env.RUN_DISPLAY_URL}|Open>)\n- ${env.GIT_COMMIT_DESC} (<https://github.com/commaai/one/commit/${env.GIT_COMMIT}|${env.GIT_COMMIT_SHORT}> on ${env.GIT_BRANCH} by ${env.CHANGE_AUTHOR})")
    }
  }
}