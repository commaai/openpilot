import os
from openpilot.system.hardware.hw import Paths
from openpilot.system.loggerd.uploader import UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE

for folder in os.walk(Paths.log_root()):
  for file1 in folder[2]:
    full_path = os.path.join(folder[0], file1)
    os.setxattr(full_path, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
