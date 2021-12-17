#!/usr/bin/env python3
import sys
from common.xattr import removexattr
from selfdrive.loggerd.uploader import UPLOAD_ATTR_NAME

for fn in sys.argv[1:]:
  print(f"unmarking {fn}")
  removexattr(fn, UPLOAD_ATTR_NAME)
