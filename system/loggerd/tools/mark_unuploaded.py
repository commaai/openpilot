#!/usr/bin/env python3
import os
import sys
from openpilot.system.loggerd.uploader import UPLOAD_ATTR_NAME

for fn in sys.argv[1:]:
  print(f"unmarking {fn}")
  os.removexattr(fn, UPLOAD_ATTR_NAME)
