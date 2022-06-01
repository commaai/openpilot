def phone(String ip, String step_label, String cmd) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    def ssh_cmd = """
ssh -tt -o StrictHostKeyChecking=no -i ${key_file} 'comma@${ip}' /usr/bin/bash <<'END'

set -e

export CI=1
export TEST_DIR=${env.TEST_DIR}
export SOURCE_DIR=${env.SOURCE_DIR}
export GIT_BRANCH=${env.GIT_BRANCH}
export GIT_COMMIT=${env.GIT_COMMIT}

source ~/.bash_profile
if [ -f /TICI ]; then
  source /etc/profile
fi

ln -snf ${env.TEST_DIR} /data/pythonpath

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
    TEST_DIR = "/data/openpilot"
    SOURCE_DIR = "/data/openpilot_source/"
  }
  options {
    timeout(time: 4, unit: 'HOURS')
  }

  stages {
    stage('build release3') {
      agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
      when {
        branch 'devel-staging'
      }
      steps {
        phone_steps("tici", [
          ["build release3-staging & dashcam3-staging", "PUSH=1 $SOURCE_DIR/release/build_release.sh"],
        ])
      }
    }

    stage('openpilot tests') {
      when {
        not {
          anyOf {
            branch 'master-ci'; branch 'devel'; branch 'devel-staging';
            branch 'release2'; branch 'release2-staging'; branch 'dashcam'; branch 'dashcam-staging';
            branch 'release3'; branch 'release3-staging'; branch 'dashcam3'; branch 'dashcam3-staging';
            branch 'testing-closet*'; branch 'hotfix-*'
          }
        }
      }

      stage('simulator') {
        agent { docker { image 'ghcr.io/commaai/openpilot-base'; args '--user=root' } }
        steps {
          sh 'git config --global --add safe.directory ${WORKSPACE}'
            sh 'git submodule --quiet foreach --recursive \'git config --global --add safe.directory ${WORKSPACE}/$name\''
            sh 'git submodule update --init --recursive --force --depth 1'
            sh "cd ${WORKSPACE}/tools/sim && CI=1 ./start_carla.sh &"
            sh "cd ${WORKSPACE}/tools/sim && MOUNT_OPENPILOT=1 CI=1 ./start_openpilot_docker.sh"
        }
      }

      stages {
        stage('On-device Tests') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          stages {
            stage('parallel tests') {
              parallel {
                stage('build') {
                  environment {
                    R3_PUSH = "${env.BRANCH_NAME == 'master' ? '1' : ' '}"
                  }
                  steps {
                    phone_steps("tici", [
                      ["build master-ci", "cd $SOURCE_DIR/release && TARGET_DIR=$TEST_DIR EXTRA_FILES='tools/' ./build_devel.sh"],
                      ["build openpilot", "cd selfdrive/manager && ./build.py"],
                      ["test manager", "python selfdrive/manager/test/test_manager.py"],
                      ["onroad tests", "cd selfdrive/test/ && ./test_onroad.py"],
                      ["test car interfaces", "cd selfdrive/car/tests/ && ./test_car_interfaces.py"],
                    ])
                  }
                }

                stage('HW + Unit Tests') {
                  steps {
                    phone_steps("tici2", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["test power draw", "python selfdrive/hardware/tici/test_power_draw.py"],
                      ["test boardd loopback", "python selfdrive/boardd/tests/test_boardd_loopback.py"],
                      ["test loggerd", "python selfdrive/loggerd/tests/test_loggerd.py"],
                      ["test encoder", "LD_LIBRARY_PATH=/usr/local/lib python selfdrive/loggerd/tests/test_encoder.py"],
                      ["test sensord", "python selfdrive/sensord/test/test_sensord.py"],
                    ])
                  }
                }

                stage('camerad') {
                  steps {
                    phone_steps("tici-party", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["test camerad", "python selfdrive/camerad/test/test_camerad.py"],
                      ["test exposure", "python selfdrive/camerad/test/test_exposure.py"],
                    ])
                  }
                }

                stage('replay') {
                  steps {
                    phone_steps("tici3", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["model replay", "cd selfdrive/test/process_replay && ./model_replay.py"],
                    ])
                  }
                }

                stage('simulator') {
                  steps {
                    dir("tools/sim/test") {
                      sh "PYTHONPATH=:${WORKSPACE} CI=1 ./test_carla_integration.py"
                    }
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
