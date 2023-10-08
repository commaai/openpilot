def device(String ip, String step_label, String cmd) {
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

def deviceStage(String stageName, String deviceType, List env, def steps) {
  stage(stageName) {
    if (currentBuild.result != null) {
        return
    }

    def extra = env.collect { "export ${it}" }.join('\n');

    docker.image('ghcr.io/commaai/alpine-ssh').inside('--user=root') {
      lock(resource: "", label: deviceType, inversePrecedence: true, variable: 'device_ip', quantity: 1) {
        timeout(time: 20, unit: 'MINUTES') {
          device(device_ip, "git checkout", extra + "\n" + readFile("selfdrive/test/setup_device_ci.sh"))
          steps.each { item ->
            device(device_ip, item[0], item[1])
          }
        }
      }
    }
  }
}

def pcStage(String stageName, Closure body) {
  node {
  stage(stageName) {
    if (currentBuild.result != null) {
        return
    }

    checkout scm

    def dockerArgs = '--user=batman -v /tmp/comma_download_cache:/tmp/comma_download_cache -v /tmp/scons_cache:/tmp/scons_cache';
    docker.build("openpilot-base:build-${env.GIT_COMMIT}", "-f Dockerfile.openpilot_base .").inside(dockerArgs) {
      timeout(time: 20, unit: 'MINUTES') {
        try {
          // TODO: remove these after all jenkins jobs are running as batman (merged with master)
          sh "sudo chown -R batman:batman /tmp/scons_cache"
          sh "sudo chown -R batman:batman /tmp/comma_download_cache"

          sh "git config --global --add safe.directory '*'"
          sh "git submodule update --init --recursive"
          sh "git lfs pull"
          body()
        } finally {
          sh "rm -rf ${env.WORKSPACE}/* || true"
          sh "rm -rf .* || true"
        }
      }
    }
  }
  }
}

def setupCredentials() {
  withCredentials([
    string(credentialsId: 'azure_token', variable: 'AZURE_TOKEN'),
    string(credentialsId: 'mapbox_token', variable: 'MAPBOX_TOKEN')
  ]) {
    env.AZURE_TOKEN = "${AZURE_TOKEN}"
    env.MAPBOX_TOKEN = "${MAPBOX_TOKEN}"
  }
}

node {
  env.CI = "1"
  env.PYTHONWARNINGS = "error"
  env.TEST_DIR = "/data/openpilot"
  env.SOURCE_DIR = "/data/openpilot_source/"
  setupCredentials()

  env.GIT_BRANCH = checkout(scm).GIT_BRANCH
  env.GIT_COMMIT = checkout(scm).GIT_COMMIT

  def excludeBranches = ['master-ci', 'devel', 'devel-staging', 'release3', 'release3-staging',
                         'dashcam3', 'dashcam3-staging', 'testing-closet*', 'hotfix-*']
  def excludeRegex = excludeBranches.join('|').replaceAll('\\*', '.*')

  if (env.BRANCH_NAME != 'master') {
    properties([
        disableConcurrentBuilds(abortPrevious: true)
    ])
  }

  try {
    if (env.BRANCH_NAME == 'devel-staging') {
      deviceStage("build release3-staging", "tici-needs-can", [], [
        ["build release3-staging & dashcam3-staging", "RELEASE_BRANCH=release3-staging DASHCAM_BRANCH=dashcam3-staging $SOURCE_DIR/release/build_release.sh"],
      ])
    }

    if (env.BRANCH_NAME == 'master-ci') {
      deviceStage("build nightly", "tici-needs-can", [], [
        ["build nightly", "RELEASE_BRANCH=nightly $SOURCE_DIR/release/build_release.sh"],
      ])
    }

    if (!env.BRANCH_NAME.matches(excludeRegex)) {
    parallel (
      // tici tests
      'onroad tests': {
        deviceStage("onroad", "tici-needs-can", ["SKIP_COPY=1"], [
          ["build master-ci", "cd $SOURCE_DIR/release && TARGET_DIR=$TEST_DIR ./build_devel.sh"],
          ["build openpilot", "cd selfdrive/manager && ./build.py"],
          ["check dirty", "release/check-dirty.sh"],
          ["onroad tests", "cd selfdrive/test/ && ./test_onroad.py"],
          ["time to onroad", "cd selfdrive/test/ && pytest test_time_to_onroad.py"],
        ])
      },
      'HW + Unit Tests': {
        deviceStage("tici", "tici-common", ["UNSAFE=1"], [
          ["build", "cd selfdrive/manager && ./build.py"],
          ["test pandad", "pytest selfdrive/boardd/tests/test_pandad.py"],
          ["test power draw", "./system/hardware/tici/tests/test_power_draw.py"],
          ["test encoder", "LD_LIBRARY_PATH=/usr/local/lib pytest system/loggerd/tests/test_encoder.py"],
          ["test pigeond", "pytest system/sensord/tests/test_pigeond.py"],
          ["test manager", "pytest selfdrive/manager/test/test_manager.py"],
          ["test nav", "pytest selfdrive/navd/tests/"],
        ])
      },
      'loopback': {
        deviceStage("tici", "tici-loopback", ["UNSAFE=1"], [
          ["build openpilot", "cd selfdrive/manager && ./build.py"],
          ["test boardd loopback", "pytest selfdrive/boardd/tests/test_boardd_loopback.py"],
        ])
      },
      'camerad': {
        deviceStage("AR0231", "tici-ar0231", ["UNSAFE=1"], [
          ["build", "cd selfdrive/manager && ./build.py"],
          ["test camerad", "pytest system/camerad/test/test_camerad.py"],
          ["test exposure", "pytest system/camerad/test/test_exposure.py"],
        ])
        deviceStage("OX03C10", "tici-ox03c10", ["UNSAFE=1"], [
          ["build", "cd selfdrive/manager && ./build.py"],
          ["test camerad", "pytest system/camerad/test/test_camerad.py"],
          ["test exposure", "pytest system/camerad/test/test_exposure.py"],
        ])
      },
      'sensord': {
        deviceStage("LSM + MMC", "tici-lsmc", ["UNSAFE=1"], [
          ["build", "cd selfdrive/manager && ./build.py"],
          ["test sensord", "cd system/sensord/tests && pytest test_sensord.py"],
        ])
        deviceStage("BMX + LSM", "tici-bmx-lsm", ["UNSAFE=1"], [
          ["build", "cd selfdrive/manager && ./build.py"],
          ["test sensord", "cd system/sensord/tests && pytest test_sensord.py"],
        ])
      },
      'replay': {
        deviceStage("tici", "tici-replay", ["UNSAFE=1"], [
          ["build", "cd selfdrive/manager && ./build.py"],
          ["model replay", "cd selfdrive/test/process_replay && ./model_replay.py"],
        ])
      },
      'tizi': {
        deviceStage("tizi", "tizi", ["UNSAFE=1"], [
          ["build openpilot", "cd selfdrive/manager && ./build.py"],
          ["test boardd loopback", "SINGLE_PANDA=1 pytest selfdrive/boardd/tests/test_boardd_loopback.py"],
          ["test pandad", "pytest selfdrive/boardd/tests/test_pandad.py"],
          ["test amp", "pytest system/hardware/tici/tests/test_amplifier.py"],
          ["test hw", "pytest system/hardware/tici/tests/test_hardware.py"],
          ["test rawgpsd", "pytest system/sensord/rawgps/test_rawgps.py"],
        ])
      },

      // *** PC tests ***
      'PC tests': {
        pcStage("PC tests") {
          // tests that our build system's dependencies are configured properly,
          // needs a machine with lots of cores
          sh label: "test multi-threaded build", script: "scons --no-cache --random -j42"
        }
      },
      'car tests': {
        pcStage("car tests") {
          sh "scons -j30"
          sh label: "test_models.py", script: "INTERNAL_SEG_CNT=250 INTERNAL_SEG_LIST=selfdrive/car/tests/test_models_segs.txt FILEREADER_CACHE=1 \
              pytest -n42 --dist=loadscope selfdrive/car/tests/test_models.py"
          sh label: "test_car_interfaces.py", script: "MAX_EXAMPLES=100 pytest -n42 selfdrive/car/tests/test_car_interfaces.py"
        }
      },

    )
    }
  } catch (Exception e) {
    currentBuild.result = 'FAILED'
    throw e
  }
}