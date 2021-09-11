def phone(String ip, String step_label, String cmd) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    def ssh_cmd = """
ssh -tt -o StrictHostKeyChecking=no -i ${key_file} -p 8022 'comma@${ip}' /usr/bin/bash <<'EOF'

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

EOF"""

    sh script: ssh_cmd, label: step_label
  }
}

def phone_steps(String device_type, steps) {
  lock(resource: "", label: device_type, inversePrecedence: true, variable: 'device_ip', quantity: 1) {
    timeout(time: 150, unit: 'MINUTES') {
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
      timeout(time: 3, unit: 'HOURS')
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
          ["build release2-staging & dashcam-staging", "cd release && PUSH=1 ./build_release2.sh"],
        ])
      }
    }

    stage('Build release3') {
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
        phone_steps("tici", [
          ["build release3-staging & dashcam3-staging", "PUSH=1 $SOURCE_DIR/release/build_release3.sh"],
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
              /*
              filename 'Dockerfile.ondevice_ci'
              args "--privileged -v /dev:/dev --shm-size=1G --user=root"
              */
              image 'python:3.7.3'
              args '--user=root'
            }
          }

          stages {
            stage('parallel tests') {
              parallel {
                stage('Devel Tests') {
                  steps {
                    phone_steps("eon-build", [
                      ["build devel", "cd $SOURCE_DIR/release && EXTRA_FILES='tools/' ./build_devel.sh"],
                      ["build openpilot", "cd selfdrive/manager && ./build.py"],
                      ["test manager", "python selfdrive/manager/test/test_manager.py"],
                      ["onroad tests", "cd selfdrive/test/ && ./test_onroad.py"],
                      ["test car interfaces", "cd selfdrive/car/tests/ && ./test_car_interfaces.py"],
                    ])
                  }
                }

                stage('Replay Tests') {
                  steps {
                    phone_steps("eon2", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["model replay", "cd selfdrive/test/process_replay && ./model_replay.py"],
                    ])
                  }
                }

                stage('HW + Unit Tests') {
                  steps {
                    phone_steps("eon", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["test athena", "nosetests -s selfdrive/athena/tests/test_athenad_old.py"],
                      ["test sounds", "nosetests -s selfdrive/ui/tests/test_sounds.py"],
                      ["test boardd loopback", "nosetests -s selfdrive/boardd/tests/test_boardd_loopback.py"],
                      ["test loggerd", "python selfdrive/loggerd/tests/test_loggerd.py"],
                      ["test encoder", "python selfdrive/loggerd/tests/test_encoder.py"],
                      ["test logcatd", "python selfdrive/logcatd/tests/test_logcatd_android.py"],
                      ["test updater", "python selfdrive/hardware/eon/test_neos_updater.py"],
                    ])
                  }
                }

                /*
                stage('Power Consumption Tests') {
                  steps {
                    lock(resource: "", label: "c2-zookeeper", inversePrecedence: true, variable: 'device_ip', quantity: 1) {
                      timeout(time: 90, unit: 'MINUTES') {
                        sh script: "/home/batman/tools/zookeeper/enable_and_wait.py $device_ip 120", label: "turn on device"
                        phone(device_ip, "git checkout", readFile("selfdrive/test/setup_device_ci.sh"),)
                        phone(device_ip, "build", "scons -j4 && sync")
                        sh script: "/home/batman/tools/zookeeper/disable.py $device_ip", label: "turn off device"
                        sh script: "/home/batman/tools/zookeeper/enable_and_wait.py $device_ip 120", label: "turn on device"
                        sh script: "/home/batman/tools/zookeeper/check_consumption.py 60 3", label: "idle power consumption after boot"
                        sh script: "/home/batman/tools/zookeeper/ignition.py 1", label: "go onroad"
                        sh script: "/home/batman/tools/zookeeper/check_consumption.py 60 10", label: "onroad power consumption"
                        sh script: "/home/batman/tools/zookeeper/ignition.py 0", label: "go offroad"
                        sh script: "/home/batman/tools/zookeeper/check_consumption.py 60 2", label: "idle power consumption offroad"
                      }
                    }
                  }
                }
                */

                stage('tici Build') {
                  environment {
                    R3_PUSH = "${env.BRANCH_NAME == 'master' ? '1' : ' '}"
                  }
                  steps {
                    phone_steps("tici", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["onroad tests", "cd selfdrive/test/ && ./test_onroad.py"],
                    ])
                  }
                }

                stage('Unit Tests (tici)') {
                  steps {
                    phone_steps("tici2", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["test loggerd", "python selfdrive/loggerd/tests/test_loggerd.py"],
                      ["test encoder", "LD_LIBRARY_PATH=/usr/local/lib python selfdrive/loggerd/tests/test_encoder.py"],
                    ])
                  }
                }

                stage('camerad') {
                  steps {
                    phone_steps("eon-party", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["test camerad", "python selfdrive/camerad/test/test_camerad.py"],
                      ["test exposure", "python selfdrive/camerad/test/test_exposure.py"],
                    ])
                  }
                }

                stage('Tici camerad') {
                  steps {
                    phone_steps("tici-party", [
                      ["build", "cd selfdrive/manager && ./build.py"],
                      ["test camerad", "python selfdrive/camerad/test/test_camerad.py"],
                      ["test exposure", "python selfdrive/camerad/test/test_exposure.py"],
                    ])
                  }
                }

              }
            }

            stage('Push master-ci') {
              when {
                branch 'master'
              }
              steps {
                phone_steps("eon-build", [
                  ["push devel", "cd $SOURCE_DIR/release && PUSH='master-ci' ./build_devel.sh"],
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

