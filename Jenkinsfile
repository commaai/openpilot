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
  agent {
    docker {
      image 'python:3.7.3'
      args '--user=root'
    }
  }
  environment {
    COMMA_JWT = credentials('athena-test-jwt')
    TEST_DIR = "/data/openpilot"
  }

  stages {

    stage('Release Build') {
      when {
        branch 'devel-staging'
      }
      steps {
        phone_steps("eon-build", [
          ["build release2-staging and dashcam-staging", "cd release && PUSH=1 ./build_release2.sh"],
        ])
      }
    }

    stage('On-device Tests') {
      when {
        not {
          anyOf {
            branch 'master-ci'; branch 'devel'; branch 'devel-staging'; branch 'release2'; branch 'release2-staging'; branch 'dashcam'; branch 'dashcam-staging'
          }
        }
      }

      parallel {

        stage('Build') {
          environment {
            CI_PUSH = "${env.BRANCH_NAME == 'master' ? 'master-ci' : ' '}"
          }
          steps {
            phone_steps("eon", [
              ["build devel", "cd release && CI_PUSH=${env.CI_PUSH} ./build_devel.sh"],
              ["test openpilot", "nosetests -s selfdrive/test/test_openpilot.py"],
              //["test cpu usage", "cd selfdrive/test/ && ./test_cpu_usage.py"],
              ["test car interfaces", "cd selfdrive/car/tests/ && ./test_car_interfaces.py"],
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

        stage('HW Tests') {
          steps {
            phone_steps("eon", [
              ["build cereal", "SCONS_CACHE=1 scons -j4 cereal/"],
              ["test sounds", "nosetests -s selfdrive/test/test_sounds.py"],
              ["test boardd loopback", "nosetests -s selfdrive/boardd/tests/test_boardd_loopback.py"],
            ])
          }
        }

      }
    }

  }
}
