import subprocess
# from common.basedir import BASEDIR
BASEDIR = "/home/shane/openpilot"
import os
env = os.environ.copy()
env['SCONS_PROGRESS'] = "1"

# scons = subprocess.Popen(["scons", "-j4", "--cache-populate"], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)
scons = subprocess.Popen(["git", "clone", "https://github.com/commaai/openpilot", "/home/shane/tmppilot", "--progress"], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)

compile_output = []

# Read progress from stderr and update spinner
while scons.poll() is None:
  # try:
  line = scons.stderr.readline()
  if line is None:
    continue
  print(f"--->>> {line}")
  print('---LINE END---')

