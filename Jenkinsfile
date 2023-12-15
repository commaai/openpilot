def retryWithDelay(int maxRetries, int delay, Closure body) {
  for (int i = 0; i < maxRetries; i++) {
    try {
      return body()
    } catch (Exception e) {
      sleep(delay)
    }
  }
  throw Exception("Failed after ${maxRetries} retries")
}

def device(String ip, Closure body) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    sh label: ssh_setup, script:


    body()
  }

  def alpine_ssh = retryWithDelay (3, 15) {
    return docker.image('ghcr.io/commaai/alpine-ssh')
  }
}

def deviceStage(String stageName, String deviceType, List env, Closure body) {
  stage(stageName) {
    if (currentBuild.result != null) {
        return
    }

    def extra = env.collect { "export ${it}" }.join('\n');

    lock(resource: "", label: deviceType, inversePrecedence: true, variable: 'device_ip', quantity: 1) {
      timeout(time: 20, unit: 'MINUTES') {
        alpine_ssh.inside('--user=root --entrypoint=') {
          device(device_ip) {
            retry (3) {
              sh label: "git checkout", script: extra + "\n" + readFile("selfdrive/test/setup_device_ci.sh")
            }
            body()
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

    def dockerArgs = "--user=batman -v /tmp/comma_download_cache:/tmp/comma_download_cache -v /tmp/scons_cache:/tmp/scons_cache -e PYTHONPATH=${env.WORKSPACE}";

    def openpilot_base = retryWithDelay (3, 15) {
      return docker.build("openpilot-base:build-${env.GIT_COMMIT}", "-f Dockerfile.openpilot_base .")
    }
    
    openpilot_base.inside(dockerArgs) {
      timeout(time: 20, unit: 'MINUTES') {
        try {
          retryWithDelay (3, 15) {
            sh "git config --global --add safe.directory '*'"
            sh "git submodule update --init --recursive"
            sh "git lfs pull"
          }
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
        deviceStage("onroad", "tici-needs-can", ["SKIP_COPY=1"], {
          sh label: "build master-ci", script: "cd $SOURCE_DIR/release && TARGET_DIR=$TEST_DIR $SOURCE_DIR/scripts/retry.sh ./build_devel.sh",
          sh label: "build openpilot", script: "cd selfdrive/manager && ./build.py",
          sh label: "check dirty", script: "release/check-dirty.sh",
          sh label: "onroad tests", script: "pytest selfdrive/test/test_onroad.py -s",
          sh label: "time to onroad", script: "pytest selfdrive/test/test_time_to_onroad.py",
        })
      },
      'HW + Unit Tests': {
        deviceStage("tici", "tici-common", ["UNSAFE=1"], {
          sh label: "build", script: "cd selfdrive/manager && ./build.py",
          sh label: "test pandad", script: "pytest selfdrive/boardd/tests/test_pandad.py",
          sh label: "test power draw", script: "./system/hardware/tici/tests/test_power_draw.py",
          sh label: "test encoder", script: "LD_LIBRARY_PATH=/usr/local/lib pytest system/loggerd/tests/test_encoder.py",
          sh label: "test pigeond", script: "pytest system/sensord/tests/test_pigeond.py",
          sh label: "test manager", script: "pytest selfdrive/manager/test/test_manager.py",
        })
      },
      'loopback': {
        deviceStage("tici", "tici-loopback", ["UNSAFE=1"], {
          sh label: "build openpilot", script: "cd selfdrive/manager && ./build.py",
          sh label: "test boardd loopback", script: "pytest selfdrive/boardd/tests/test_boardd_loopback.py",
        })
      },
      'camerad': {
        deviceStage("AR0231", "tici-ar0231", ["UNSAFE=1"], {
          sh label: "build", script: "cd selfdrive/manager && ./build.py",
          sh label: "test camerad", script: "pytest system/camerad/test/test_camerad.py",
          sh label: "test exposure", script: "pytest system/camerad/test/test_exposure.py",
        }),
        deviceStage("OX03C10", "tici-ox03c10", ["UNSAFE=1"], {
          sh label: "build", script: "cd selfdrive/manager && ./build.py",
          sh label: "test camerad", script: "pytest system/camerad/test/test_camerad.py",
          sh label: "test exposure", script: "pytest system/camerad/test/test_exposure.py",
        })
      },
      'sensord': {
        deviceStage("LSM + MMC", "tici-lsmc", ["UNSAFE=1"], {
          sh label: "build", script: "cd selfdrive/manager && ./build.py",
          sh label: "test sensord", script: "pytest system/sensord/tests/test_sensord.py",
        }),
        deviceStage("BMX + LSM", "tici-bmx-lsm", ["UNSAFE=1"], {
          sh label: "build", script: "cd selfdrive/manager && ./build.py",
          sh label: "test sensord", script: "pytest system/sensord/tests/test_sensord.py",
        })
      },
      'replay': {
        deviceStage("tici", "tici-replay", ["UNSAFE=1"], {
          sh label: "build", script: "cd selfdrive/manager && ./build.py",
          sh label: "model replay", script: "selfdrive/test/process_replay/model_replay.py",
        })
      },
      'tizi': {
        deviceStage("tizi", "tizi", ["UNSAFE=1"]) {
          sh label: "build", script: "cd selfdrive/manager && ./build.py",
          sh label: "test boardd loopback", script: "SINGLE_PANDA=1 pytest selfdrive/boardd/tests/test_boardd_loopback.py",
          sh label: "test pandad", script: "pytest selfdrive/boardd/tests/test_pandad.py",
          sh label: "test amp", script: "pytest system/hardware/tici/tests/test_amplifier.py",
          sh label: "test hw", script: "pytest system/hardware/tici/tests/test_hardware.py",
          sh label: "test qcomgpsd", script: "pytest system/qcomgpsd/tests/test_qcomgpsd.py",
        }
      },

      // *** PC tests ***
      'PC tests': {
        pcStage("PC tests") {
          // tests that our build system's dependencies are configured properly,
          // needs a machine with lots of cores
          sh label: "test multi-threaded build",
             script: '''#!/bin/bash
                        scons --no-cache --random -j$(nproc)'''
        }
      },
      'car tests': {
        pcStage("car tests") {
          sh label: "build", script: "selfdrive/manager/build.py"
          sh label: "run car tests", script: "cd selfdrive/car/tests && MAX_EXAMPLES=100 INTERNAL_SEG_CNT=250 FILEREADER_CACHE=1 \
              INTERNAL_SEG_LIST=selfdrive/car/tests/test_models_segs.txt pytest test_models.py test_car_interfaces.py"
        }
      },

    )
    }
  } catch (Exception e) {
    currentBuild.result = 'FAILED'
    throw e
  }
}