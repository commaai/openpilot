def phone(String ip, String step_label, String cmd) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    def ssh_cmd = """
ssh -tt -o StrictHostKeyChecking=no -i ${key_file} 'comma@${ip}' /usr/bin/bash <<'END'

set -e

export CI=1
export PYTHONWARNINGS=error
export LOGPRINT=debug
export TEST_DIR=${env.TEST_DIR}
export SOURCE_DIR=${env.SOURCE_DIR}
export GIT_BRANCH=${env.GIT_BRANCH}
export GIT_COMMIT=${env.GIT_COMMIT}
export AZURE_TOKEN='${env.AZURE_TOKEN}'
export MAPBOX_TOKEN='${env.MAPBOX_TOKEN}'

export GIT_SSH_COMMAND="ssh -i /data/gitkey"

source ~/.bash_profile
if [ -f /TICI ]; then
  source /etc/profile

  if ! systemctl is-active --quiet systemd-resolved; then
    echo "restarting resolved"
    sudo systemctl start systemd-resolved
    sleep 3
  fi

  # restart aux USB
  if [ -e /sys/bus/usb/drivers/hub/3-0:1.0 ]; then
    echo "restarting aux usb"
    echo "3-0:1.0" | sudo tee /sys/bus/usb/drivers/hub/unbind
    sleep 0.5
    echo "3-0:1.0" | sudo tee /sys/bus/usb/drivers/hub/bind
  fi
fi
if [ -f /data/openpilot/launch_env.sh ]; then
  source /data/openpilot/launch_env.sh
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
    CI = "1"
    PYTHONWARNINGS = "error"
    TEST_DIR = "/data/openpilot"
    SOURCE_DIR = "/data/openpilot_source/"
    AZURE_TOKEN = credentials('azure_token')
    MAPBOX_TOKEN = credentials('mapbox_token')
  }
  options {
    timeout(time: 3, unit: 'HOURS')
    disableConcurrentBuilds(abortPrevious: env.BRANCH_NAME != 'master')
  }

  stages {
    stage('build release3-staging') {
      agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
      when {
        branch 'devel-staging'
      }
      steps {
        phone_steps("tici-needs-can", [
          ["build release3-staging & dashcam3-staging", "RELEASE_BRANCH=release3-staging DASHCAM_BRANCH=dashcam3-staging $SOURCE_DIR/release/build_release.sh"],
        ])
      }
    }

    stage('build nightly') {
      agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
      when {
        branch 'master-ci'
      }
      steps {
        phone_steps("tici-needs-can", [
          ["build nightly", "RELEASE_BRANCH=nightly $SOURCE_DIR/release/build_release.sh"],
        ])
      }
    }

    stage('openpilot tests') {
      when {
        not {
          anyOf {
            branch 'master-ci'; branch 'devel'; branch 'devel-staging';
            branch 'release3'; branch 'release3-staging'; branch 'dashcam3'; branch 'dashcam3-staging';
            branch 'testing-closet*'; branch 'hotfix-*'
          }
        }
      }

      parallel {

        /*
        stage('simulator') {
          agent {
            dockerfile {
              filename 'Dockerfile.sim_nvidia'
              dir 'tools/sim'
              args '--user=root'
            }
          }
          steps {
            sh "git config --global --add safe.directory '*'"
            sh "git submodule update --init --recursive"
            sh "git lfs pull"
            lock(resource: "", label: "simulator", inversePrecedence: true, quantity: 1) {
              sh "${WORKSPACE}/tools/sim/build_container.sh"
              sh "DETACH=1 ${WORKSPACE}/tools/sim/start_carla.sh"
              sh "${WORKSPACE}/tools/sim/start_openpilot_docker.sh"
            }
          }

          post {
            always {
              sh "docker kill carla_sim || true"
              sh "rm -rf ${WORKSPACE}/* || true"
              sh "rm -rf .* || true"
            }
          }
        }
        */

        stage('PC tests') {
          agent {
            dockerfile {
              filename 'Dockerfile.openpilot_base'
              args '--user=root -v /tmp/comma_download_cache:/tmp/comma_download_cache'
            }
          }
          steps {
            sh "git config --global --add safe.directory '*'"
            sh "git submodule update --init --depth=1 --recursive"
            sh "git lfs pull"
            // tests that our build system's dependencies are configured properly, needs a machine with lots of cores
            sh "scons --clean && scons --no-cache --random -j42"
            sh "INTERNAL_SEG_CNT=500 INTERNAL_SEG_LIST=selfdrive/car/tests/test_models_segs.txt FILEREADER_CACHE=1 \
                pytest -n42 --dist=loadscope selfdrive/car/tests/test_models.py"
          }

          post {
            always {
              sh "rm -rf ${WORKSPACE}/* || true"
              sh "rm -rf .* || true"
            }
          }
        }

        stage('tizi-tests') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          steps {
            phone_steps("tizi", [
              ["build openpilot", "cd selfdrive/manager && ./build.py"],
              ["test boardd loopback", "SINGLE_PANDA=1 pytest selfdrive/boardd/tests/test_boardd_loopback.py"],
              ["test pandad", "pytest selfdrive/boardd/tests/test_pandad.py"],
              ["test sensord", "cd system/sensord/tests && pytest test_sensord.py"],
              ["test camerad", "pytest system/camerad/test/test_camerad.py"],
              ["test exposure", "pytest system/camerad/test/test_exposure.py"],
              ["test amp", "pytest system/hardware/tici/tests/test_amplifier.py"],
              ["test hw", "pytest system/hardware/tici/tests/test_hardware.py"],
              ["test rawgpsd", "pytest system/sensord/rawgps/test_rawgps.py"],
            ])
          }
        }

        stage('build') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          environment {
            R3_PUSH = "${env.BRANCH_NAME == 'master' ? '1' : ' '}"
          }
          steps {
            phone_steps("tici-needs-can", [
              ["build master-ci", "cd $SOURCE_DIR/release && TARGET_DIR=$TEST_DIR ./build_devel.sh"],
              ["build openpilot", "cd selfdrive/manager && ./build.py"],
              ["check dirty", "release/check-dirty.sh"],
              ["onroad tests", "cd selfdrive/test/ && ./test_onroad.py"],
              ["time to onroad", "cd selfdrive/test/ && pytest test_time_to_onroad.py"],
            ])
          }
        }

        stage('loopback-tests') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          steps {
            phone_steps("tici-loopback", [
              ["build openpilot", "cd selfdrive/manager && ./build.py"],
              ["test boardd loopback", "pytest selfdrive/boardd/tests/test_boardd_loopback.py"],
            ])
          }
        }

        stage('HW + Unit Tests') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          steps {
            phone_steps("tici-common", [
              ["build", "cd selfdrive/manager && ./build.py"],
              ["test pandad", "pytest selfdrive/boardd/tests/test_pandad.py"],
              ["test power draw", "pytest system/hardware/tici/tests/test_power_draw.py"],
              ["test encoder", "LD_LIBRARY_PATH=/usr/local/lib pytest system/loggerd/tests/test_encoder.py"],
              ["test pigeond", "pytest system/sensord/tests/test_pigeond.py"],
              ["test manager", "pytest selfdrive/manager/test/test_manager.py"],
              ["test nav", "pytest selfdrive/navd/tests/"],
            ])
          }
        }

        stage('camerad') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          steps {
            phone_steps("tici-ar0231", [
              ["build", "cd selfdrive/manager && ./build.py"],
              ["test camerad", "pytest system/camerad/test/test_camerad.py"],
              ["test exposure", "pytest system/camerad/test/test_exposure.py"],
            ])
            phone_steps("tici-ox03c10", [
              ["build", "cd selfdrive/manager && ./build.py"],
              ["test camerad", "pytest system/camerad/test/test_camerad.py"],
              ["test exposure", "pytest system/camerad/test/test_exposure.py"],
            ])
          }
        }

        stage('sensord') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          steps {
            phone_steps("tici-lsmc", [
              ["build", "cd selfdrive/manager && ./build.py"],
              ["test sensord", "cd system/sensord/tests && pytest test_sensord.py"],
            ])
            phone_steps("tici-bmx-lsm", [
              ["build", "cd selfdrive/manager && ./build.py"],
              ["test sensord", "cd system/sensord/tests && pytest test_sensord.py"],
            ])
          }
        }

        stage('replay') {
          agent { docker { image 'ghcr.io/commaai/alpine-ssh'; args '--user=root' } }
          steps {
            phone_steps("tici-replay", [
              ["build", "cd selfdrive/manager && ./build.py"],
              ["model replay", "cd selfdrive/test/process_replay && ./model_replay.py"],
            ])
          }
        }

      }
    }

  }
}
