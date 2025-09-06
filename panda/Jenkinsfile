def docker_run(String step_label, int timeout_mins, String cmd) {
  timeout(time: timeout_mins, unit: 'MINUTES') {
    sh script: "docker run --rm --privileged \
          --env PYTHONWARNINGS=error \
          --volume /dev/bus/usb:/dev/bus/usb \
          --volume /var/run/dbus:/var/run/dbus \
          --net host \
          ${env.DOCKER_IMAGE_TAG} \
          bash -c 'scons -j8 && ${cmd}'", \
        label: step_label
  }
}


def phone(String ip, String step_label, String cmd) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    def ssh_cmd = """
ssh -tt -o StrictHostKeyChecking=no -i ${key_file} 'comma@${ip}' /usr/bin/bash <<'END'

set -e


source ~/.bash_profile
if [ -f /etc/profile ]; then
  source /etc/profile
fi

export CI=1
export TEST_DIR=${env.TEST_DIR}
export SOURCE_DIR=${env.SOURCE_DIR}
export GIT_BRANCH=${env.GIT_BRANCH}
export GIT_COMMIT=${env.GIT_COMMIT}
export PYTHONPATH=${env.TEST_DIR}/../
export PYTHONWARNINGS=error
ln -sf /data/openpilot/opendbc_repo/opendbc /data/opendbc

# TODO: this is an agnos issue
export PYTEST_ADDOPTS="-p no:asyncio"

cd ${env.TEST_DIR} || true
${cmd}
exit 0

END"""

    sh script: ssh_cmd, label: step_label
  }
}

def phone_steps(String device_type, steps) {
  lock(resource: "", label: device_type, inversePrecedence: true, variable: 'device_ip', quantity: 1) {
    timeout(time: 20, unit: 'MINUTES') {
      phone(device_ip, "git checkout", readFile("tests/setup_device_ci.sh"),)
      steps.each { item ->
        phone(device_ip, item[0], item[1])
      }
    }
  }
}



pipeline {
  agent any
  environment {
    CI = "1"
    PYTHONWARNINGS= "error"
    DOCKER_IMAGE_TAG = "panda:build-${env.GIT_COMMIT}"

    TEST_DIR = "/data/panda"
    SOURCE_DIR = "/data/panda_source/"
  }
  options {
    timeout(time: 3, unit: 'HOURS')
    disableConcurrentBuilds(abortPrevious: env.BRANCH_NAME != 'master')
  }

  stages {
    stage ('Acquire resource locks') {
      options {
        lock(resource: "pandas")
      }
      stages {
        stage('Build Docker Image') {
          steps {
            timeout(time: 20, unit: 'MINUTES') {
              script {
                dockerImage = docker.build("${env.DOCKER_IMAGE_TAG}", "--build-arg CACHEBUST=${env.BUILD_NUMBER} .")
              }
            }
          }
        }
        stage('jungle tests') {
          steps {
            script {
              retry (3) {
                docker_run("reset hardware", 3, "python3 ./tests/hitl/reset_jungles.py")
              }
            }
          }
        }

        stage('parallel tests') {
          parallel {
            stage('test cuatro') {
              agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
              steps {
                phone_steps("panda-cuatro", [
                  ["build", "scons -j4"],
                  ["flash", "cd scripts/ && ./reflash_internal_panda.py"],
                  ["flash jungle", "cd board/jungle && ./flash.py --all"],
                  ["test", "cd tests/hitl && HW_TYPES=10 pytest --durations=0 2*.py [5-9]*.py"],
                ])
              }
            }

            stage('test tres') {
              agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
              steps {
                phone_steps("panda-tres", [
                  ["build", "scons -j4"],
                  ["flash", "cd scripts/ && ./reflash_internal_panda.py"],
                  ["flash jungle", "cd board/jungle && ./flash.py --all"],
                  ["test", "cd tests/hitl && HW_TYPES=9 pytest --durations=0 2*.py [5-9]*.py"],
                ])
              }
            }

            /*
            stage('bootkick tests') {
              steps {
                script {
                  docker_run("test", 10, "pytest ./tests/som/test_bootkick.py")
                }
              }
            }
            */
          }
        }
      }
    }
  }
}
