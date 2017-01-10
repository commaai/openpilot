import os

# fetch from environment
def get_dongle_id_and_secret():
  return os.getenv("DONGLE_ID"), os.getenv("DONGLE_SECRET") 

ROOT = '/sdcard/realdata/'

SEGMENT_LENGTH = 60
