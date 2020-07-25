def phone(String ip, String cmd, String step_label) {
  def ci_env = "CI=1 TEST_DIR=${env.TEST_DIR} GIT_BRANCH=${env.GIT_BRANCH} GIT_COMMIT=${env.GIT_COMMIT}"

  withCredentials([file(credentialsId: 'id_rsa_public', variable: 'key_file')]) {
    sh label: step_label,
        script: """
                ssh -o StrictHostKeyChecking=no -i ${key_file} -p 8022 root@${ip} '${ci_env} /usr/bin/bash -xle' <<'EOF'
                cd ${env.TEST_DIR} || true
                ${cmd}
EOF"""
  }
}

def phone_steps(String device_type, int timeout, steps) {
  lock(resource: "", label: device_type, inversePrecedence: true, variable: 'device_ip', quantity: 1) {
    timeout(time: timeout, unit: 'MINUTES') {
      phone(ip, "pkill -f comma && pkill -f selfdrive", "kill old processes")
      phone(ip, readFile("selfdrive/test/setup_device_ci.sh"), "git checkout")
      steps.each { item ->
        phone(device_ip, item[0], item[1])
      }
    }
  }
}

pipeline {
  agent {
    docker {
      image 'ubuntu:latest'
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
        phone_steps("eon-build", 60, [["cd release && PUSH=1 ./build_release2.sh", "build release2-staging and dashcam-staging"]])
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
            CI_PUSH = "${env.BRANCH_NAME == 'master' ? 'master-ci' : ''}"
          }
          steps {
            phone_steps("eon", 60, [["cd release && CI_PUSH=${env.CI_PUSH} ./build_devel.sh", "build devel"]])
          }
        }

        stage('Replay Tests') {
          steps {
            phone_steps("eon2", 60, [["cd selfdrive/test/process_replay && ./camera_replay.py", "camerad/modeld replay"]])
          }
        }

        stage('HW Tests') {
          steps {
            phone_steps("eon", 60, [
              ["SCONS_CACHE=1 scons -j4 cereal/", "build cereal"],
              ["nosetests -s selfdrive/test/test_sounds.py", "test sounds"],
              ["nosetests -s selfdrive/boardd/tests/test_boardd_loopback.py", "test boardd loopback"],
            ])
          }
        }

      }
    }

  }
}
