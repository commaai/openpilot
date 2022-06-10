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

if [ -f /EON ]; then
  echo \$\$ > /dev/cpuset/app/tasks || true
  echo \$PPID > /dev/cpuset/app/tasks || true
  mkdir -p /dev/shm
  chmod 777 /dev/shm
fi

cd ${env.TEST_DIR} || true
${cmd}
exit 0

END"""

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
    TEST_DIR = "/data/openpilot"
    SOURCE_DIR = "/data/openpilot_source/"
  }
  options {
      timeout(time: 4, unit: 'HOURS')
  }

  stages {
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

      stages {
        stage('On-device Tests') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          stages {
            stage('parallel tests') {
              parallel {
                stage('C2: build') {
                  steps {
                    phone_steps("eon-build", [
                      ["build master-ci", "cd $SOURCE_DIR/release && EXTRA_FILES='tools/' ./build_devel.sh"],
                      ["build openpilot", "cd selfdrive/manager && ./build.py"],
                      ["test manager", "python selfdrive/manager/test/test_manager.py"],
                      ["onroad tests", "cd selfdrive/test/ && ./test_onroad.py"],
                      ["test car interfaces", "cd selfdrive/car/tests/ && ./test_car_interfaces.py"],
                    ])
                  }
                }

                stage('C2: replay') {
                  steps {
                    phone_steps("eon2", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["model replay", "cd selfdrive/test/process_replay && ./model_replay.py"],
                    ])
                  }
                }

                stage('C2: HW + Unit Tests') {
                  steps {
                    phone_steps("eon", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["test sounds", "python selfdrive/ui/tests/test_soundd.py"],
                      ["test boardd loopback", "python selfdrive/boardd/tests/test_boardd_loopback.py"],
                      ["test loggerd", "python selfdrive/loggerd/tests/test_loggerd.py"],
                      ["test encoder", "python selfdrive/loggerd/tests/test_encoder.py"],
                      ["test logcatd", "python selfdrive/logcatd/tests/test_logcatd_android.py"],
                      ["test updater", "python selfdrive/hardware/eon/test_neos_updater.py"],
                    ])
                  }
                }

              }
            }

            stage('Push release2-staging') {
              when {
                branch 'commatwo_master'
              }
              steps {
                phone_steps("eon-build", [
                  ["build release2-staging", "PUSH=1 $SOURCE_DIR/release/build_release.sh"],
                ])
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

