import os
import subprocess
from selfdrive.swaglog import cloudlog

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common", "version.h")) as _versionf:
  version = _versionf.read().split('"')[1]

try:
  origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).rstrip()
  if origin.startswith('git@github.com:commaai') or origin.startswith('https://github.com/commaai'):
    if origin.endswith('/one.git'):
      dirty = True
    else:
      branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).rstrip()
      branch = 'origin/' + branch
      subprocess.check_call(["git", "update-index", "--refresh"])  #This is needed otherwise touched files might show up as modified
      dirty = subprocess.call(["git", "diff-index", "--quiet", branch, "--"]) != 0
      if dirty:
        dirty_files = subprocess.check_output(["git", "diff-index", branch, "--"])
        commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"]).rstrip()
        origin_commit = subprocess.check_output(["git", "rev-parse", "--verify", branch]).rstrip()
        cloudlog.event("dirty comma branch", vesion=version, dirty=dirty, origin=origin, branch=branch, dirty_files=dirty_files, commit=commit, origin_commit=origin_commit)
  else:
    dirty = True
except subprocess.CalledProcessError:
  try:
    cloudlog.exception("git subprocess failed while finding version")
  except:
    pass
  dirty = True

# put this here
training_version = "0.1.0"
