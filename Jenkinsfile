def phone(String ip, String step_label, String cmd) {
  withCredentials([file(credentialsId: 'id_rsa_public', variable: 'key_file')]) {
    def ssh_cmd = """
ssh -tt -o StrictHostKeyChecking=no -i ${key_file} -p 8022 'comma@${ip}' /usr/bin/bash <<'EOF'

set -e

export CI=1
export TEST_DIR=${env.TEST_DIR}
export GIT_BRANCH=${env.GIT_BRANCH}
export GIT_COMMIT=${env.GIT_COMMIT}

source ~/.bash_profile

ln -snf ${env.TEST_DIR} /data/pythonpath

if [ -f /EON ]; then
  echo \$\$ > /dev/cpuset/app/tasks || true
  echo \$PPID > /dev/cpuset/app/tasks || true
  mkdir -p /dev/shm
  chmod 777 /dev/shm
fi

cd ${env.TEST_DIR} || true
${cmd}
exit 0

EOF"""

    sh script: ssh_cmd, label: step_label
  }
}

def phone_steps(String device_type, steps) {
  lock(resource: "", label: device_type, inversePrecedence: true, variable: 'device_ip', quantity: 1) {
    timeout(time: 60, unit: 'MINUTES') {
      phone(device_ip, "git checkout", readFile("selfdrive/test/setup_device_ci.sh"),)
      steps.each { item ->
        phone(device_ip, item[0], item[1])
      }
    }
  }
}

pipeline {
  agent none
  environment {
    COMMA_JWT = credentials('athena-test-jwt')
    TEST_DIR = "/data/openpilot"
  }
  options {
      timeout(time: 1, unit: 'HOURS')
  }

  stages {

    stage('Build release2') {
      agent {
        docker {
          image 'python:3.7.3'
          args '--user=root'
        }
      }
      when {
        branch 'devel-staging'
      }
      steps {
        phone_steps("eon-build", [
          ["build release2-staging and dashcam-staging", "cd release && PUSH=1 ./build_release2.sh"],
        ])
      }
    }

    stage('openpilot tests') {
      when {
        not {
          anyOf {
            branch 'master-ci'; branch 'devel'; branch 'devel-staging'; branch 'release2'; branch 'release2-staging'; branch 'dashcam'; branch 'dashcam-staging'; branch 'testing-closet*'
          }
        }
      }


      stages {

        /*
        stage('PC tests') {
          agent {
            dockerfile {
              filename 'Dockerfile.openpilotci'
              args '--privileged --shm-size=1G --user=root'
            }
          }
          stages {
            stage('Build') {
              steps {
                sh 'scons -j$(nproc)'
              }
            }
          }
          post {
            always {
              // fix permissions since docker runs as another user
              sh "chmod -R 777 ."
            }
          }
        }
        */

        stage('On-device Tests') {
          agent {
            docker {
              image 'python:3.7.3'
              args '--user=root'
            }
          }

          stages {
            stage('parallel tests') {
              parallel {
                stage('Devel Build') {
                  environment {
                    CI_PUSH = "${env.BRANCH_NAME == 'master' ? 'master-ci' : ' '}"
                  }
                  steps {
                    phone_steps("eon-build", [
                      ["build", "SCONS_CACHE=1 scons -j4"],
                      ["test athena", "nosetests -s selfdrive/athena/tests/test_athenad_old.py"],
                      ["test manager", "python selfdrive/manager/test/test_manager.py"],
                      ["onroad tests", "cd selfdrive/test/ && ./test_onroad.py"],
                      ["build devel", "cd release && CI_PUSH=${env.CI_PUSH} ./build_devel.sh"],
                      ["test car interfaces", "cd selfdrive/car/tests/ && ./test_car_interfaces.py"],
                      ["test spinner build", "cd selfdrive/ui/spinner && make clean && make"],
                      ["test text window build", "cd selfdrive/ui/text && make clean && make"],
                    ])
                  }
                }

                stage('Replay Tests') {
                  steps {
                    phone_steps("eon2", [
                      ["build QCOM_REPLAY", "SCONS_CACHE=1 QCOM_REPLAY=1 scons -j4"],
                      ["camerad/modeld replay", "cd selfdrive/test/process_replay && ./camera_replay.py"],
                    ])
                  }
                }

                stage('HW + Unit Tests') {
                  steps {
                    phone_steps("eon", [
                      ["build", "SCONS_CACHE=1 scons -j4"],
                      ["test sounds", "nosetests -s selfdrive/test/test_sounds.py"],
                      ["test boardd loopback", "nosetests -s selfdrive/boardd/tests/test_boardd_loopback.py"],
                      ["test loggerd", "python selfdrive/loggerd/tests/test_loggerd.py"],
                      ["test encoder", "python selfdrive/loggerd/tests/test_encoder.py"],
                      ["test camerad", "python selfdrive/camerad/test/test_camerad.py"],
                      ["test logcatd", "python selfdrive/logcatd/tests/test_logcatd_android.py"],
                      //["test updater", "python installer/updater/test_updater.py"],
                    ])
                  }
                }

                stage('Tici Build') {
                  environment {
                    R3_PUSH = "${env.BRANCH_NAME == 'master' ? '1' : ' '}"
                  }
                  steps {
                    phone_steps("tici", [
                      ["build", "SCONS_CACHE=1 scons -j16"],
                      ["test loggerd", "python selfdrive/loggerd/tests/test_loggerd.py"],
                      ["test encoder", "LD_LIBRARY_PATH=/usr/local/lib python selfdrive/loggerd/tests/test_encoder.py"],
                      ["test camerad", "python selfdrive/camerad/test/test_camerad.py"],
                      //["build release3-staging", "cd release && PUSH=${env.R3_PUSH} ./build_release3.sh"],
                    ])
                  }
                }

              }
            }
          }

          post {
            always {
              cleanWs()
            }
          }

        }

      }
    }
  }
}

