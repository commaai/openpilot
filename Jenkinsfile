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

def device(String ip, String step_label, String cmd) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    def ssh_cmd = """
ssh -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o BatchMode=yes -o StrictHostKeyChecking=no -i ${key_file} 'comma@${ip}' exec /usr/bin/bash <<'END'

set -e

export TERM=xterm-256color

shopt -s huponexit # kill all child processes when the shell exits

export CI=1
export PYTHONWARNINGS=error
export LOGPRINT=debug
export TEST_DIR=${env.TEST_DIR}
export SOURCE_DIR=${env.SOURCE_DIR}
export GIT_BRANCH=${env.GIT_BRANCH}
export GIT_COMMIT=${env.GIT_COMMIT}
export CI_ARTIFACTS_TOKEN=${env.CI_ARTIFACTS_TOKEN}
export GITHUB_COMMENTS_TOKEN=${env.GITHUB_COMMENTS_TOKEN}
export AZURE_TOKEN='${env.AZURE_TOKEN}'
# only use 1 thread for tici tests since most require HIL
export PYTEST_ADDOPTS="-n 0"


export GIT_SSH_COMMAND="ssh -i /data/gitkey"

source ~/.bash_profile
if [ -f /TICI ]; then
  source /etc/profile

  rm -rf /tmp/tmp*
  rm -rf ~/.commacache
  rm -rf /dev/shm/*
  rm -rf /dev/tmp/tmp*

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
time ${cmd}
END"""

    sh script: ssh_cmd, label: step_label
  }
}

def deviceStage(String stageName, String deviceType, List extra_env, def steps) {
  stage(stageName) {
    if (currentBuild.result != null) {
        return
    }

    def extra = extra_env.collect { "export ${it}" }.join('\n');
    def branch = env.BRANCH_NAME ?: 'master';
    def gitDiff = sh returnStdout: true, script: 'curl -s -H "Authorization: Bearer ${GITHUB_COMMENTS_TOKEN}" https://api.github.com/repos/commaai/openpilot/compare/master...${GIT_BRANCH} | jq .files[].filename || echo "/"', label: 'Getting changes'

    lock(resource: "", label: deviceType, inversePrecedence: true, variable: 'device_ip', quantity: 1, resourceSelectStrategy: 'random') {
      docker.image('ghcr.io/commaai/alpine-ssh').inside('--user=root') {
        timeout(time: 35, unit: 'MINUTES') {
          retry (3) {
            def date = sh(script: 'date', returnStdout: true).trim();
            device(device_ip, "set time", "date -s '" + date + "'")
            device(device_ip, "git checkout", extra + "\n" + readFile("selfdrive/test/setup_device_ci.sh"))
          }
          steps.each { item ->
            def name = item[0]
            def cmd = item[1]

            def args = item[2]
            def argPaths = args.diffPaths ?: []
            def argTimeout = args.timeout ?: 300

            if (branch != "master" && argPaths && !hasPathChanged(gitDiff, argPaths)) {
              println "Skipping ${name}: no changes in ${argPaths}."
              return
            } else {
              timeout(time: argTimeout, unit: 'SECONDS') {
                device(device_ip, name, cmd)
              }
            }
          }
        }
      }
    }
  }
}

def hasPathChanged(String gitDiff, List<String> paths) {
  for (path in paths) {
    if (gitDiff.contains(path)) {
      return true
    }
  }
  return false
}

def step(String name, String cmd, Map args = [:]) {
  return [name, cmd, args]
}

def setupCredentials() {
  withCredentials([
    string(credentialsId: 'azure_token', variable: 'AZURE_TOKEN'),
  ]) {
    env.AZURE_TOKEN = "${AZURE_TOKEN}"
  }

  withCredentials([
    string(credentialsId: 'ci_artifacts_pat', variable: 'CI_ARTIFACTS_TOKEN'),
  ]) {
    env.CI_ARTIFACTS_TOKEN = "${CI_ARTIFACTS_TOKEN}"
  }

  withCredentials([
    string(credentialsId: 'post_comments_github_pat', variable: 'GITHUB_COMMENTS_TOKEN'),
  ]) {
    env.GITHUB_COMMENTS_TOKEN = "${GITHUB_COMMENTS_TOKEN}"
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
                         'testing-closet*', 'hotfix-*']
  def excludeRegex = excludeBranches.join('|').replaceAll('\\*', '.*')

  if (env.BRANCH_NAME != 'master') {
    properties([
        disableConcurrentBuilds(abortPrevious: true)
    ])
  }

  try {
    if (env.BRANCH_NAME == 'devel-staging') {
      deviceStage("build release3-staging", "tici-needs-can", [], [
        step("build release3-staging", "RELEASE_BRANCH=release3-staging $SOURCE_DIR/release/build_release.sh"),
      ])
    }

    if (env.BRANCH_NAME == 'master-ci') {
      deviceStage("build nightly", "tici-needs-can", [], [
        step("build nightly", "RELEASE_BRANCH=nightly $SOURCE_DIR/release/build_release.sh"),
      ])
    }

    if (!env.BRANCH_NAME.matches(excludeRegex)) {
    parallel (
      'model replay': {
        deviceStage("model replay", "tici-replay", ["UNSAFE=1"], [
          step("build", "cd system/manager && ./build.py", [diffPaths: ["selfdrive/modeld/"]]),
          step("model replay", "selfdrive/test/process_replay/model_replay.py", [diffPaths: ["selfdrive/modeld/"], timeout: 65]),
        ])
      },
      'onroad tests': {
        deviceStage("onroad tests", "tici-needs-can", ["UNSAFE=1"], [
          step("build openpilot", "cd system/manager && ./build.py"),
          step("check dirty", "release/check-dirty.sh"),
          step("onroad tests", "pytest selfdrive/test/test_onroad.py -s", [timeout: 65]),
        ])
      },
    )
    }
  } catch (Exception e) {
    currentBuild.result = 'FAILED'
    throw e
  }
}
