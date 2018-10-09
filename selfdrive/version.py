import os
import subprocess
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
      dirty = subprocess.call(["git", "diff-index", "--quiet", branch, "--"]) != 0
  else:
    dirty = True
except subprocess.CalledProcessError:
  dirty = True

# put this here
training_version = "0.1.0"
