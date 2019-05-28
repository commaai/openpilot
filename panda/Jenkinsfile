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
            sh 'git clone --no-checkout --depth 1 git@github.com:commaai/xx.git || true'
            sh 'cd xx && git fetch origin && git checkout origin/master -- pandaextra && cd ..' // Needed for certs for panda flashing
            sh 'git archive -v -o panda.tar.gz --format=tar.gz HEAD'
            dockerImage = docker.build("${env.DOCKER_IMAGE_TAG}")
          }
        }
      }
    }
    stage('Test Dev Build') {
      steps {
        lock(resource: "Pandas", inversePrecedence: true, quantity:1){
          timeout(time: 60, unit: 'MINUTES') {
            sh "docker run --name ${env.DOCKER_NAME} --privileged --volume /dev/bus/usb:/dev/bus/usb --volume /var/run/dbus:/var/run/dbus --net host ${env.DOCKER_IMAGE_TAG} bash -c 'cd /tmp/panda; ./run_automated_tests.sh '"
          }
        }
      }
    }
    stage('Test EON Build') {
      steps {
        lock(resource: "Pandas", inversePrecedence: true, quantity:1){
          timeout(time: 60, unit: 'MINUTES') {
            sh "docker cp ${env.DOCKER_NAME}:/tmp/panda/nosetests.xml test_results_dev.xml"
            sh "touch EON && docker cp EON ${env.DOCKER_NAME}:/EON"
            sh "docker start -a ${env.DOCKER_NAME}"
          }
        }
      }
    }
  }
  post {
    always {
      script {
        sh "docker cp ${env.DOCKER_NAME}:/tmp/panda/nosetests.xml test_results_EON.xml"
        sh "docker rm ${env.DOCKER_NAME}"
      }
      junit "test_results*.xml"
    }
  }
}