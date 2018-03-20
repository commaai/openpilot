import os
import subprocess
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common", "version.h")) as _versionf:
  version = _versionf.read().split('"')[1]

try:
  origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
  if "-private" in origin:
    upstream = "origin/master"
  else:
    if 'chffrplus' in origin:
      upstream = "origin/release"
    else:
      upstream = "origin/release2"

  dirty = subprocess.call(["git", "diff-index", "--quiet", upstream, "--"]) != 0
except subprocess.CalledProcessError:
  dirty = True
