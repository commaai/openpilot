pipeline {
  agent any
  environment {
    DOCKER_IMAGE_TAG = "panda:build-${env.GIT_COMMIT}"
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
    stage('PEDAL tests') {
      steps {
        lock(resource: "pedal", inversePrecedence: true, quantity: 1) {
          timeout(time: 10, unit: 'MINUTES') {
            script {
              sh "docker run --rm --privileged \
                    --volume /dev/bus/usb:/dev/bus/usb \
                    --volume /var/run/dbus:/var/run/dbus \
                    --net host \
                    ${env.DOCKER_IMAGE_TAG} \
                    bash -c 'cd /tmp/panda && PEDAL_JUNGLE=058010800f51363038363036 python ./tests/pedal/test_pedal.py'"
            }
          }
        }
      }
    }
    stage('CANFD tests') {
      steps {
        lock(resource: "pedal", inversePrecedence: true, quantity: 1) {
          timeout(time: 10, unit: 'MINUTES') {
            script {
              sh "docker run --rm --privileged \
                    --volume /dev/bus/usb:/dev/bus/usb \
                    --volume /var/run/dbus:/var/run/dbus \
                    --net host \
                    ${env.DOCKER_IMAGE_TAG} \
                    bash -c 'cd /tmp/panda && ./board/build_all.sh && JUNGLE=058010800f51363038363036 H7_PANDAS_EXCLUDE=\"080021000c51303136383232\" python ./tests/canfd/test_ci_canfd.py'"
            }
          }
        }
      }
    }
    stage('HITL tests') {
      steps {
        lock(resource: "pandas", inversePrecedence: true, quantity: 1) {
          timeout(time: 20, unit: 'MINUTES') {
            script {
              sh "docker run --rm --privileged \
                    --volume /dev/bus/usb:/dev/bus/usb \
                    --volume /var/run/dbus:/var/run/dbus \
                    --net host \
                    ${env.DOCKER_IMAGE_TAG} \
                    bash -c 'cd /tmp/panda && ./board/build_all.sh && PANDAS_JUNGLE=23002d000851393038373731 PANDAS_EXCLUDE=\"1d0002000c51303136383232 2f002e000c51303136383232\" ./tests/automated/test.sh'"
            }
          }
        }
      }
    }
  }
}
