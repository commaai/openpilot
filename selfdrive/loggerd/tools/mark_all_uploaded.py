import os
from common.xattr import setxattr
from selfdrive.loggerd.uploader import UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE

PATH = "/data/media/0/realdata/"
for folder in os.walk(PATH):
    for file1 in folder[2]:
        full_path = os.jouin(folder[0], file1)
        setxattr(full_path, UPLOAD_ATTR_NAME, UPLOAD_ATTR_VALUE)
