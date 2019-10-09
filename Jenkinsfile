pipeline {
  agent any
  environment {
    AUTHOR = """${sh(
                returnStdout: true,
                script: "git --no-pager show -s --format='%an' ${GIT_COMMIT}"
             ).trim()}"""

    DOCKER_IMAGE_TAG = "panda:build-${env.GIT_COMMIT}"
    DOCKER_NAME = "panda-test-${env.GIT_COMMIT}"
  }
  stages {
    stage('Build Docker Image') {
      steps {
        timeout(time: 60, unit: 'MINUTES') {
          script {
            sh 'git archive -v -o panda.tar.gz --format=tar.gz HEAD'
            dockerImage = docker.build("${env.DOCKER_IMAGE_TAG}")
          }
        }
      }
    }
    stage('Test Dev Build (no WIFI)') {
      steps {
        lock(resource: "Pandas", inversePrecedence: true, quantity: 1){
          timeout(time: 60, unit: 'MINUTES') {
            script {
              sh "docker run --name ${env.DOCKER_NAME} --privileged --volume /dev/bus/usb:/dev/bus/usb --volume /var/run/dbus:/var/run/dbus --net host ${env.DOCKER_IMAGE_TAG} bash -c 'cd /tmp/panda; SKIPWIFI=1 ./run_automated_tests.sh'"
              sh "docker cp ${env.DOCKER_NAME}:/tmp/panda/nosetests.xml test_results_dev_nowifi.xml"
              sh "docker rm ${env.DOCKER_NAME}"
            }
          }
        }
      }
    }
    stage('Test EON Build') {
      steps {
        lock(resource: "Pandas", inversePrecedence: true, quantity: 1){
          timeout(time: 60, unit: 'MINUTES') {
            script {
              sh "docker run --name ${env.DOCKER_NAME} --privileged --volume /dev/bus/usb:/dev/bus/usb --volume /var/run/dbus:/var/run/dbus --net host ${env.DOCKER_IMAGE_TAG} bash -c 'touch /EON; cd /tmp/panda; ./run_automated_tests.sh'"
              sh "docker cp ${env.DOCKER_NAME}:/tmp/panda/nosetests.xml test_results_eon.xml"
              sh "docker rm ${env.DOCKER_NAME}"
            }
          }
        }
      }
    }
  }
  post {
    failure {
      script {
        sh "docker rm ${env.DOCKER_NAME} || true"
      }
    }
    always {
      junit "test_results*.xml"
    }
  }
}
