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

def device(String ip, String step_label, String cmd, String controlPath = "") {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    def ssh_cmd = """
${rsyncSshCommand(key_file, controlPath)} 'comma@${ip}' exec /usr/bin/bash <<'END'

set -e

export TERM=xterm-256color

shopt -s huponexit # kill all child processes when the shell exits

export CI=1
export PYTHONWARNINGS=error
#export LOGPRINT=debug # this has gotten too spammy...
export TEST_DIR=${env.TEST_DIR}
export SOURCE_DIR=${env.SOURCE_DIR}
export GIT_BRANCH=${env.GIT_BRANCH}
export GIT_COMMIT=${env.GIT_COMMIT}
export CI_ARTIFACTS_TOKEN=${env.CI_ARTIFACTS_TOKEN}
export GITHUB_COMMENTS_TOKEN=${env.GITHUB_COMMENTS_TOKEN}
export AZURE_TOKEN='${env.AZURE_TOKEN}'
# only use 1 thread for tici tests since most require HIL
export PYTEST_ADDOPTS="-n0 -s"


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

def installRsync() {
  sh script: '''
set -e

if ! command -v rsync >/dev/null; then
  mkdir -p /device-build-cache/apk
  apk --cache-dir /device-build-cache/apk add --update-cache rsync
fi
''', label: 'install rsync'
}

def ciDockerArgs() {
  return '--user=root -v /var/jenkins_home/openpilot-device-build-cache:/device-build-cache'
}

def cleanOldWorkspaceBuildCache() {
  docker.image('ghcr.io/commaai/alpine-ssh').inside(ciDockerArgs()) {
    sh script: "rm -rf '${env.WORKSPACE}/.device-build-cache'", label: 'clean old workspace build cache'
  }
}

def rsyncSshCommand(String keyFile, String controlPath = "") {
  def controlArgs = controlPath ? " -o ControlMaster=no -o ControlPath=${controlPath}" : ""
  return "ssh -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o BatchMode=yes -o StrictHostKeyChecking=no -i ${keyFile}${controlArgs}"
}

def sshControlPath(String ip) {
  return "/tmp/openpilot-ci-ssh-${ip.replaceAll('[^A-Za-z0-9_.-]', '_')}"
}

def startSshMaster(String ip, String controlPath) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    sh script: """
set -e
rm -f '${controlPath}'
${rsyncSshCommand(key_file)} -o ControlMaster=yes -o ControlPath='${controlPath}' -o ControlPersist=120s 'comma@${ip}' -Nf
""", label: 'start ssh master'
  }
}

def stopSshMaster(String ip, String controlPath) {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    sh script: """
${rsyncSshCommand(key_file)} -o ControlPath='${controlPath}' -O exit 'comma@${ip}' >/dev/null 2>&1 || true
rm -f '${controlPath}'
""", label: 'stop ssh master'
  }
}

def rsyncBuiltTreeFromDevice(String ip, String controlPath = "") {
  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    sh script: """
set -e

cache='${env.DEVICE_BUILD_DIR}'
mkdir -p "\${cache}"
rm -rf "\${cache}/.git" "\${cache}/.venv" "\${cache}/.ruff_cache" "\${cache}/.pytest_cache" "\${cache}/.mypy_cache"
rm -f "\${cache}/.sconsign.dblite"

old_manifest="\${cache}/.ci_manifest"
old_id="\$(cat "\${cache}/.ci_manifest.id" 2>/dev/null || true)"
manifest_dir="\${cache}/.ci_manifests"
mkdir -p "\${manifest_dir}"
if [ -n "\${old_id}" ] && [ -f "\${old_manifest}" ]; then
  cp "\${old_manifest}" "\${manifest_dir}/\${old_id}"
fi

new_manifest="\$(mktemp)"
new_id_file="\$(mktemp)"
sync_paths="\${cache}/.ci_sync_paths"
changed_paths="\${cache}/.ci_changed_paths"
deleted_paths="\${cache}/.ci_deleted_paths"
cache_updated=0
trap 'rm -f "\${new_manifest}" "\${new_id_file}"' EXIT

ssh_cmd='${rsyncSshCommand(key_file, controlPath)}'
ssh_cmd="\${ssh_cmd} comma@${ip}"

\${ssh_cmd} "cat '${env.TEST_DIR}/.ci_manifest.id'" > "\${new_id_file}"
new_id="\$(cat "\${new_id_file}")"
printf '%s\\n' "\${old_id}" > "\${cache}/.ci_previous_manifest.id"

if [ -f "\${old_manifest}" ] && [ "\${old_id}" = "\${new_id}" ]; then
  : > "\${changed_paths}"
  : > "\${deleted_paths}"
  : > "\${sync_paths}"
  echo "builder cache manifest unchanged: \${new_id}"
elif [ -f "\${old_manifest}" ]; then
  \${ssh_cmd} "cat '${env.TEST_DIR}/.ci_manifest'" > "\${new_manifest}"
  old_entries="\$(mktemp)"
  new_entries="\$(mktemp)"
  old_paths="\$(mktemp)"
  new_paths="\$(mktemp)"
  trap 'rm -f "\${new_manifest}" "\${new_id_file}" "\${old_entries}" "\${new_entries}" "\${old_paths}" "\${new_paths}"' EXIT
  awk -F '\\t' 'BEGIN { OFS = "\\t" } { print \$3, \$1, \$2 }' "\${old_manifest}" | sort > "\${old_entries}"
  awk -F '\\t' 'BEGIN { OFS = "\\t" } { print \$3, \$1, \$2 }' "\${new_manifest}" | sort > "\${new_entries}"
  cut -f1 "\${old_entries}" > "\${old_paths}"
  cut -f1 "\${new_entries}" > "\${new_paths}"
  comm -13 "\${old_entries}" "\${new_entries}" | cut -f1 > "\${changed_paths}"
  comm -23 "\${old_paths}" "\${new_paths}" > "\${deleted_paths}"
  cat "\${changed_paths}" "\${deleted_paths}" > "\${sync_paths}"
  printf '.ci_manifest\\n.ci_manifest.id\\n' >> "\${sync_paths}"
  echo "changed=\$(wc -l < "\${changed_paths}") deleted=\$(wc -l < "\${deleted_paths}")"
  sed -n '1,40p' "\${changed_paths}"
  rsync -a --delete-missing-args --no-owner --no-group --info=stats2,name0 \\
    --ignore-times \\
    --files-from="\${sync_paths}" \\
    -e '${rsyncSshCommand(key_file, controlPath)}' \\
    'comma@${ip}:${env.TEST_DIR}/' "\${cache}/"
  cache_updated=1
else
  \${ssh_cmd} "cat '${env.TEST_DIR}/.ci_manifest'" > "\${new_manifest}"
  echo "builder cache has no manifest, doing full content sync"
  rsync -a --delete --delete-excluded --checksum --no-owner --no-group --info=stats2,name0 \\
    --exclude='.git' --exclude='.git/' --exclude='.git/**' \\
    --exclude='.venv' --exclude='.ruff_cache' --exclude='.pytest_cache' --exclude='.mypy_cache' \\
    --exclude='__pycache__' --exclude='.sconsign.dblite' \\
    -e '${rsyncSshCommand(key_file, controlPath)}' \\
    'comma@${ip}:${env.TEST_DIR}/' "\${cache}/"
  find "\${cache}" -depth -type d -empty -delete
  : > "\${sync_paths}"
  : > "\${changed_paths}"
  : > "\${deleted_paths}"
  cache_updated=1
fi

if [ "\${cache_updated}" = "1" ]; then
  cp "\${new_manifest}" "\${cache}/.ci_manifest"
  cp "\${new_id_file}" "\${cache}/.ci_manifest.id"
  cp "\${new_manifest}" "\${manifest_dir}/\${new_id}"
else
  cp "\${old_manifest}" "\${manifest_dir}/\${new_id}"
fi
printf '%s\\n' "\${new_id}" > "\${cache}/.ci_current_manifest.id"
echo "builder cache previous manifest: \${old_id:-none}"
echo "builder cache current manifest: \${new_id}"
echo "builder cache changed paths: \$(wc -l < "\${changed_paths}")"
echo "builder cache deleted paths: \$(wc -l < "\${deleted_paths}")"
""", label: 'cache built tree'
  }

  env.DEVICE_BUILD_PREVIOUS_ID = sh(script: "cat '${env.DEVICE_BUILD_DIR}/.ci_previous_manifest.id' 2>/dev/null || true", returnStdout: true).trim()
  env.DEVICE_BUILD_ID = sh(script: "cat '${env.DEVICE_BUILD_DIR}/.ci_current_manifest.id'", returnStdout: true).trim()
}

def rsyncBuiltTreeToDevice(String ip, String controlPath = "") {
  if (env.DEVICE_BUILD_READY != "1") {
    return
  }

  withCredentials([file(credentialsId: 'id_rsa', variable: 'key_file')]) {
    sh script: """
set -e

cache='${env.DEVICE_BUILD_DIR}'
previous_id='${env.DEVICE_BUILD_PREVIOUS_ID ?: ''}'
current_id='${env.DEVICE_BUILD_ID ?: ''}'
ssh_cmd='${rsyncSshCommand(key_file, controlPath)}'
remote="comma@${ip}"
remote_manifest=""

remote_id="\$(\${ssh_cmd} "\${remote}" "cat '${env.TEST_DIR}/.ci_manifest.id' 2>/dev/null || true")"
if [ -n "\${remote_id}" ] && [ -f "\${cache}/.ci_manifests/\${remote_id}" ]; then
  remote_manifest="\${cache}/.ci_manifests/\${remote_id}"
fi

if [ -n "\${current_id}" ] && [ "\${remote_id}" = "\${current_id}" ]; then
  echo "device already has manifest \${current_id}"
elif [ -n "\${remote_id}" ] && [ -f "\${remote_manifest}" ]; then
  echo "delta sync from \${remote_id} to \${current_id}"
  \${ssh_cmd} "\${remote}" "mkdir -p '${env.TEST_DIR}' && rm -rf '${env.TEST_DIR}/.git' '${env.TEST_DIR}/.venv' '${env.TEST_DIR}/.ruff_cache' '${env.TEST_DIR}/.pytest_cache' '${env.TEST_DIR}/.mypy_cache' && rm -f '${env.TEST_DIR}/.sconsign.dblite'"
  current_manifest="\${cache}/.ci_manifest"
  old_entries="\$(mktemp)"
  new_entries="\$(mktemp)"
  old_paths="\$(mktemp)"
  new_paths="\$(mktemp)"
  sync_paths="\$(mktemp)"
  changed_paths="\$(mktemp)"
  deleted_paths="\$(mktemp)"
  trap 'rm -f "\${old_entries}" "\${new_entries}" "\${old_paths}" "\${new_paths}" "\${sync_paths}" "\${changed_paths}" "\${deleted_paths}"' EXIT
  awk -F '\\t' 'BEGIN { OFS = "\\t" } { print \$3, \$1, \$2 }' "\${remote_manifest}" | sort > "\${old_entries}"
  awk -F '\\t' 'BEGIN { OFS = "\\t" } { print \$3, \$1, \$2 }' "\${current_manifest}" | sort > "\${new_entries}"
  cut -f1 "\${old_entries}" > "\${old_paths}"
  cut -f1 "\${new_entries}" > "\${new_paths}"
  comm -13 "\${old_entries}" "\${new_entries}" | cut -f1 > "\${changed_paths}"
  comm -23 "\${old_paths}" "\${new_paths}" > "\${deleted_paths}"
  cat "\${changed_paths}" "\${deleted_paths}" > "\${sync_paths}"
  printf '.ci_manifest\\n.ci_manifest.id\\n' >> "\${sync_paths}"
  echo "changed=\$(wc -l < "\${changed_paths}") deleted=\$(wc -l < "\${deleted_paths}")"
  sed -n '1,40p' "\${changed_paths}"
  if [ -s "\${sync_paths}" ]; then
    rsync -a --delete-missing-args --no-owner --no-group --info=stats2,name0 \\
      --ignore-times \\
      --files-from="\${sync_paths}" \\
      -e '${rsyncSshCommand(key_file, controlPath)}' \\
      "\${cache}/" "\${remote}:${env.TEST_DIR}/"
  else
    echo "no changed paths to sync"
  fi
else
  echo "full content sync, remote manifest was \${remote_id:-none}, expected \${previous_id:-none}"
  \${ssh_cmd} "\${remote}" "mkdir -p '${env.TEST_DIR}' && rm -rf '${env.TEST_DIR}/.git' '${env.TEST_DIR}/.venv' '${env.TEST_DIR}/.ruff_cache' '${env.TEST_DIR}/.pytest_cache' '${env.TEST_DIR}/.mypy_cache' && rm -f '${env.TEST_DIR}/.sconsign.dblite'"
  rsync -a --delete --delete-excluded --checksum --no-owner --no-group --info=stats2,name0 \\
    --exclude='.git' --exclude='.git/' --exclude='.git/**' \\
    --exclude='.venv' --exclude='.ruff_cache' --exclude='.pytest_cache' --exclude='.mypy_cache' \\
    --exclude='__pycache__' --exclude='.sconsign.dblite' \\
    --exclude='.ci_changed_paths' --exclude='.ci_deleted_paths' --exclude='.ci_sync_paths' \\
    --exclude='.ci_previous_manifest.id' --exclude='.ci_current_manifest.id' \\
    -e '${rsyncSshCommand(key_file, controlPath)}' \\
    "\${cache}/" "\${remote}:${env.TEST_DIR}/"
fi
""", label: 'sync built tree'
  }
}

def buildAndPrepareTreeCommand() {
  return """
if [ ! -f /tmp/openpilot_ci_cpu_unlocked ]; then
  for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    online="\${cpu}/online"
    if [ -w "\${online}" ]; then
      echo 1 | sudo tee "\${online}"
    fi
  done

  for policy in /sys/devices/system/cpu/cpufreq/policy*; do
    [ -d "\${policy}" ] || continue
    max="\$(cat "\${policy}/cpuinfo_max_freq")"
    echo "\${max}" | sudo tee "\${policy}/scaling_max_freq"
    echo "\${max}" | sudo tee "\${policy}/scaling_min_freq"
    if grep -qw performance "\${policy}/scaling_available_governors"; then
      echo performance | sudo tee "\${policy}/scaling_governor"
    fi
  done

  for governor in /sys/class/devfreq/soc:qcom,cpubw/governor /sys/class/devfreq/soc:qcom,memlat-cpu*/governor; do
    if [ -w "\${governor}" ]; then
      echo performance | sudo tee "\${governor}"
    fi
  done

  touch /tmp/openpilot_ci_cpu_unlocked
fi

cd system/manager
taskset -c 0-7 ./build.py

cd "\${TEST_DIR}"
dirty="\$(git status --porcelain)"
if [ -n "\${dirty}" ]; then
  echo "Dirty working tree after build:"
  echo "\${dirty}"
  exit 1
fi

python3 - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(".")

def run(cmd):
  return subprocess.check_output(cmd, text=True).strip()

metadata = {
  "channel": "${env.BRANCH_NAME ?: 'unknown'}",
  "openpilot": {
    "version": (root / "common/version.h").read_text().split('"')[1],
    "release_notes": (root / "RELEASES.md").read_text().split("\\n\\n", 1)[0],
    "git_commit": run(["git", "rev-parse", "HEAD"]),
    "git_origin": run(["git", "config", "--get", "remote.origin.url"]),
    "git_commit_date": run(["git", "show", "--no-patch", "--format=%ct %ci", "HEAD"]),
    "build_style": "ci",
  },
}
(root / "build.json").write_text(json.dumps(metadata))

raw_tracked = subprocess.check_output(["git", "ls-files", "--recurse-submodules", "-s", "-z"])
tracked = {}
for record in raw_tracked.split(b"\\0"):
  if not record:
    continue
  metadata, path = record.split(b"\\t", 1)
  mode, oid, _stage = metadata.split(b" ")
  tracked[path.decode()] = (mode.decode(), oid.decode())

entries = [("G", f"{mode}:{oid}", path) for path, (mode, oid) in tracked.items()]
skip_dirs = {".git", ".venv", ".ruff_cache", ".pytest_cache", ".mypy_cache", "__pycache__"}
skip_files = {".ci_manifest", ".ci_manifest.id", ".sconsign.dblite"}

for dirpath, dirnames, filenames in os.walk(root):
  dirpath = Path(dirpath)

  keep_dirs = []
  for name in sorted(dirnames):
    path = dirpath / name
    rel = path.relative_to(root).as_posix()
    if name in skip_dirs or rel.startswith(".git/"):
      continue
    if rel in tracked:
      continue
    if path.is_symlink():
      entries.append(("L", os.readlink(path), rel))
    else:
      keep_dirs.append(name)
  dirnames[:] = keep_dirs

  for name in sorted(filenames):
    path = dirpath / name
    rel = path.relative_to(root).as_posix()
    if rel in tracked or name == ".git" or name in skip_files:
      continue
    if path.is_symlink():
      entries.append(("L", os.readlink(path), rel))
      continue

    h = hashlib.sha256()
    with path.open("rb") as f:
      for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
    entries.append(("F", f"{path.stat().st_size}:{h.hexdigest()}", rel))

manifest = "".join(f"{kind}\\t{value}\\t{rel}\\n" for kind, value, rel in sorted(entries, key=lambda e: e[2]))
(root / ".ci_manifest").write_text(manifest)
(root / ".ci_manifest.id").write_text(hashlib.sha256(manifest.encode()).hexdigest() + "\\n")
print((root / "build.json").read_text())
print(f"manifest entries={len(entries)} id={(root / '.ci_manifest.id').read_text().strip()}")
PY
"""
}

def prepareBuiltTree() {
  stage("build device tree") {
    lock(resource: "comma-db5a74d4", inversePrecedence: true, variable: 'builder_ip') {
      docker.image('ghcr.io/commaai/alpine-ssh').inside(ciDockerArgs()) {
        timeout(time: 35, unit: 'MINUTES') {
          def controlPath = sshControlPath(builder_ip)
          try {
            startSshMaster(builder_ip, controlPath)
            retry(3) {
              def date = sh(script: 'date', returnStdout: true).trim();
              device(builder_ip, "builder git checkout", "date -s '" + date + "'\nexport UNSAFE=1\n" + readFile("selfdrive/test/setup_device_ci.sh"), controlPath)
            }
            device(builder_ip, "build and prepare tree", buildAndPrepareTreeCommand(), controlPath)
            installRsync()
            rsyncBuiltTreeFromDevice(builder_ip, controlPath)
            env.DEVICE_BUILD_READY = "1"
          } finally {
            stopSshMaster(builder_ip, controlPath)
          }
        }
      }
    }
  }
}

def deviceStage(String stageName, String deviceType, List extra_env, def steps) {
  stage(stageName) {
    if (currentBuild.result != null) {
        return
    }

    if (isReplay()) {
      error("REPLAYING TESTS IS NOT ALLOWED. FIX THEM INSTEAD.")
    }

    def extra = extra_env.collect { "export ${it}" }.join('\n');
    def branch = env.BRANCH_NAME ?: 'master';
    def gitDiff = sh returnStdout: true, script: 'curl -s -H "Authorization: Bearer ${GITHUB_COMMENTS_TOKEN}" https://api.github.com/repos/commaai/openpilot/compare/master...${GIT_BRANCH} | jq .files[].filename || echo "/"', label: 'Getting changes'

    lock(resource: "", label: deviceType, inversePrecedence: true, variable: 'device_ip', quantity: 1, resourceSelectStrategy: 'random') {
      docker.image('ghcr.io/commaai/alpine-ssh').inside(ciDockerArgs()) {
        timeout(time: 35, unit: 'MINUTES') {
          if (env.DEVICE_BUILD_READY == "1") {
            installRsync()
          }
          def controlPath = sshControlPath(device_ip)
          try {
            startSshMaster(device_ip, controlPath)
            retry (3) {
              def date = sh(script: 'date', returnStdout: true).trim();
              if (env.DEVICE_BUILD_READY == "1") {
                device(device_ip, "device setup", "date -s '" + date + "'\nmkdir -p ${env.TEST_DIR}\nexport SKIP_GIT_CHECKOUT=1\n" + readFile("selfdrive/test/setup_device_ci.sh"), controlPath)
                rsyncBuiltTreeToDevice(device_ip, controlPath)
              } else {
                device(device_ip, "git checkout", "date -s '" + date + "'\n" + extra + "\n" + readFile("selfdrive/test/setup_device_ci.sh"), controlPath)
              }
            }
            steps.each { item ->
              def name = item[0]
              def cmd = item[1]

              def args = item[2]
              def diffPaths = args.diffPaths ?: []
              def cmdTimeout = args.timeout ?: 9999

              if (env.DEVICE_BUILD_READY == "1" && (name.toLowerCase().startsWith("build") || name == "check dirty")) {
                println "Skipping ${name}: using shared built tree."
                return
              } else if (branch != "master" && !branch.contains("__jenkins_loop_") && diffPaths && !hasPathChanged(gitDiff, diffPaths)) {
                println "Skipping ${name}: no changes in ${diffPaths}."
                return
              } else {
                timeout(time: cmdTimeout, unit: 'SECONDS') {
                  device(device_ip, name, cmd, controlPath)
                }
              }
            }
          } finally {
            stopSshMaster(device_ip, controlPath)
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

def isReplay() {
  def replayClass = "org.jenkinsci.plugins.workflow.cps.replay.ReplayCause"
  return currentBuild.rawBuild.getCauses().any{ cause -> cause.toString().contains(replayClass) }
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

def step(String name, String cmd, Map args = [:]) {
  return [name, cmd, args]
}

node {
  env.CI = "1"
  env.PYTHONWARNINGS = "error"
  env.TEST_DIR = "/data/openpilot"
  env.SOURCE_DIR = "/data/openpilot_source/"
  env.DEVICE_BUILD_DIR = "/device-build-cache/${(env.BRANCH_NAME ?: 'detached').replaceAll('[^A-Za-z0-9_.-]', '_')}"
  env.DEVICE_BUILD_READY = "0"
  setupCredentials()

  cleanOldWorkspaceBuildCache()
  def scmVars = checkout(scm)
  env.GIT_BRANCH = scmVars.GIT_BRANCH
  env.GIT_COMMIT = scmVars.GIT_COMMIT

  def excludeBranches = ['__nightly', 'devel', 'devel-staging', 'release3', 'release3-staging',
                         'release-tici', 'release-tizi', 'release-tizi-staging', 'release-mici-staging', 'testing-closet*', 'hotfix-*']
  def excludeRegex = excludeBranches.join('|').replaceAll('\\*', '.*')

  if (env.BRANCH_NAME != 'master' && !env.BRANCH_NAME.contains('__jenkins_loop_')) {
    properties([
        disableConcurrentBuilds(abortPrevious: true)
    ])
  }

  try {
    if (env.BRANCH_NAME == 'devel-staging') {
      deviceStage("build release-tizi-staging", "tizi-needs-can", [], [
        step("build release-tizi-staging", "RELEASE_BRANCH=release-tizi-staging $SOURCE_DIR/release/build_release.sh && git push -f origin release-tizi-staging:release-mici-staging"),
      ])
    }

    if (env.BRANCH_NAME == '__nightly') {
      parallel (
        'nightly': {
          deviceStage("build nightly", "tizi-needs-can", [], [
            step("build nightly", "RELEASE_BRANCH=nightly $SOURCE_DIR/release/build_release.sh"),
          ])
        },
        'nightly-dev': {
          deviceStage("build nightly-dev", "tizi-needs-can", [], [
            step("build nightly-dev", "PANDA_DEBUG_BUILD=1 RELEASE_BRANCH=nightly-dev $SOURCE_DIR/release/build_release.sh"),
          ])
        },
      )
    }

    if (!env.BRANCH_NAME.matches(excludeRegex)) {
      prepareBuiltTree()
      parallel (
      'onroad tests': {
        deviceStage("onroad", "tizi-needs-can", ["UNSAFE=1"], [
          step("build openpilot", "cd system/manager && ./build.py"),
          step("check dirty", "release/check-dirty.sh"),
          step("onroad tests", "pytest selfdrive/test/test_onroad.py -s", [timeout: 60]),
        ])
      },
      'HW + Unit Tests': {
        deviceStage("tizi-hardware", "tizi-common", ["UNSAFE=1"], [
          step("build", "cd system/manager && ./build.py"),
          step("test power draw", "pytest -s system/hardware/tici/tests/test_power_draw.py"),
          step("test encoder", "LD_LIBRARY_PATH=/usr/local/lib pytest system/loggerd/tests/test_encoder.py", [diffPaths: ["system/loggerd/"]]),
          step("test manager", "pytest system/manager/test/test_manager.py"),
        ])
      },
      'camerad OX03C10': {
        deviceStage("OX03C10", "tizi-ox03c10", ["UNSAFE=1"], [
          step("build", "cd system/manager && ./build.py"),
          step("test pandad", "pytest selfdrive/pandad/tests/test_pandad.py"),
          step("test camerad", "pytest system/camerad/test/test_camerad.py", [timeout: 90]),
        ])
      },
      'camerad OS04C10': {
        deviceStage("OS04C10", "tici-os04c10", ["UNSAFE=1"], [
          step("build", "cd system/manager && ./build.py"),
          step("test pandad", "pytest selfdrive/pandad/tests/test_pandad.py"),
          step("test camerad", "pytest system/camerad/test/test_camerad.py", [timeout: 90]),
        ])
      },
      'sensord': {
        deviceStage("LSM + MMC", "tizi-lsmc", ["UNSAFE=1"], [
          step("build", "cd system/manager && ./build.py"),
          step("test sensord", "pytest system/sensord/tests/test_sensord.py"),
        ])
      },
      'replay': {
        deviceStage("model-replay", "tizi-replay", ["UNSAFE=1"], [
          step("build", "cd system/manager && ./build.py", [diffPaths: ["selfdrive/modeld/", "tinygrad_repo", "selfdrive/test/process_replay/model_replay.py"]]),
          step("model replay", "selfdrive/test/process_replay/model_replay.py", [diffPaths: ["selfdrive/modeld/", "tinygrad_repo", "selfdrive/test/process_replay/model_replay.py"]]),
        ])
      },
      'tizi': {
        deviceStage("tizi", "tizi", ["UNSAFE=1"], [
          step("build openpilot", "cd system/manager && ./build.py"),
          step("test pandad loopback", "pytest selfdrive/pandad/tests/test_pandad_loopback.py"),
          step("test pandad spi", "pytest selfdrive/pandad/tests/test_pandad_spi.py"),
          step("test amp", "pytest system/hardware/tici/tests/test_amplifier.py"),
        ])
      },

    )
    }
  } catch (Exception e) {
    currentBuild.result = 'FAILED'
    throw e
  }
}
