import os
from pathlib import Path
from openpilot.system.hardware import PC

if os.environ.get('LOG_ROOT', False):
  ROOT = os.environ['LOG_ROOT']
elif PC:
  ROOT = str(Path.home() / ".comma" / "media" / "0" / "realdata")
else:
  ROOT = '/data/media/0/realdata/'


CAMERA_FPS = 20
SEGMENT_LENGTH = 60

STATS_DIR_FILE_LIMIT = 10000
STATS_SOCKET = "ipc:///tmp/stats"
if PC:
  STATS_DIR = str(Path.home() / ".comma" / "stats")
else:
  STATS_DIR = "/data/stats/"
STATS_FLUSH_TIME_S = 60

def get_available_percent(default=None):
  try:
    statvfs = os.statvfs(ROOT)
    available_percent = 100.0 * statvfs.f_bavail / statvfs.f_blocks
  except OSError:
    available_percent = default

  return available_percent


def get_available_bytes(default=None):
  try:
    statvfs = os.statvfs(ROOT)
    available_bytes = statvfs.f_bavail * statvfs.f_frsize
  except OSError:
    available_bytes = default

  return available_bytes
