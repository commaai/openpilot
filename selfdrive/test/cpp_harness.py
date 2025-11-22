#!/usr/bin/env python3
import subprocess
import sys

from openpilot.common.prefix import OpenpilotPrefix

with OpenpilotPrefix():
  ret = subprocess.call(sys.argv[1:])

sys.exit(ret)
