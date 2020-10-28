def phone(String ip, String step_label, String cmd) {
  def ci_env = "CI=1 TEST_DIR=${env.TEST_DIR} GIT_BRANCH=${env.GIT_BRANCH} GIT_COMMIT=${env.GIT_COMMIT}"

  withCredentials([file(credentialsId: 'id_rsa_public', variable: 'key_file')]) {
    sh label: step_label,
        script: """
                ssh -tt -o StrictHostKeyChecking=no -i ${key_file} -p 8022 root@${ip} '${ci_env} /usr/bin/bash -le' <<'EOF'
echo \$\$ > /dev/cpuset/app/tasks || true
echo \$PPID > /dev/cpuset/app/tasks || true
mkdir -p /dev/shm
chmod 777 /dev/shm
cd ${env.TEST_DIR} || true
${cmd}
exit 0
EOF"""
  }
}

def phone_steps(String device_type, steps) {
  lock(resource: "", label: device_type, inversePrecedence: true, variable: 'device_ip', quantity: 1) {
    timeout(time: 60, unit: 'MINUTES') {
      phone(device_ip, "kill old processes", "pkill -f comma || true")
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

    stage('Release Build') {
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
                    phone_steps("eon", [
                      ["build devel", "cd release && CI_PUSH=${env.CI_PUSH} ./build_devel.sh"],
                      ["test openpilot", "nosetests -s selfdrive/test/test_openpilot.py"],
                      ["test cpu usage", "cd selfdrive/test/ && ./test_cpu_usage.py"],
                      ["test car interfaces", "cd selfdrive/car/tests/ && ./test_car_interfaces.py"],
                      ["test spinner build", "cd selfdrive/ui/spinner && make clean && make"],
                      ["test text window build", "cd selfdrive/ui/text && make clean && make"],
                    ])
                  }
                }

                stage('Replay Tests') {
                  steps {
                    phone_steps("eon2", [
                      ["camerad/modeld replay", "cd selfdrive/test/process_replay && ./camera_replay.py"],
                    ])
                  }
                }

                stage('HW + Unit Tests') {
                  steps {
                    phone_steps("eon", [
                      ["build cereal", "SCONS_CACHE=1 scons -j4 cereal/"],
                      ["test sounds", "nosetests -s selfdrive/test/test_sounds.py"],
                      ["test boardd loopback", "nosetests -s selfdrive/boardd/tests/test_boardd_loopback.py"],
                      ["test loggerd", "CI=1 python selfdrive/loggerd/tests/test_loggerd.py"],
                      //["test camerad", "CI=1 python selfdrive/camerad/test/test_camerad.py"], // wait for shelf refactor
                      //["test updater", "python installer/updater/test_updater.py"],
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
