import os
import platform
import sys
from pathlib import Path

from openpilot.common.hardware import PC

TMP_DIR = os.environ.get("TEMP", "/tmp")  # Path::tmp_dir()
DEFAULT_DOWNLOAD_CACHE_ROOT = "/tmp/comma_download_cache"

class Paths:
  @staticmethod
  def comma_home() -> str:
    return os.path.join(str(Path.home()), ".comma" + os.environ.get("OPENPILOT_PREFIX", ""))

  @staticmethod
  def log_root() -> str:
    if os.environ.get('LOG_ROOT', False):
      return os.environ['LOG_ROOT']
    elif PC:
      return str(Path(Paths.comma_home()) / "media" / "0" / "realdata")
    else:
      return '/data/media/0/realdata/'

  @staticmethod
  def swaglog_root() -> str:
    if PC:
      return os.path.join(Paths.comma_home(), "log")
    else:
      return "/data/log/"

  @staticmethod
  def swaglog_ipc() -> str:
    prefix = os.environ.get("OPENPILOT_PREFIX", "")
    if sys.platform == "win32":
      # libzmq has no ipc:// transport on MinGW: derive a loopback port from the prefix (FNV-1a, mirrored in hw.h)
      h = 14695981039346656037
      for c in prefix.encode():
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
      return f"tcp://127.0.0.1:{26000 + h % 1000}"
    return "ipc:///tmp/logmessage" + prefix

  @staticmethod
  def download_cache_root() -> str:
    if os.environ.get('COMMA_CACHE', False):
      return os.environ['COMMA_CACHE'] + "/"
    return DEFAULT_DOWNLOAD_CACHE_ROOT + os.environ.get("OPENPILOT_PREFIX", "") + "/"

  @staticmethod
  def persist_root() -> str:
    if PC:
      return os.path.join(Paths.comma_home(), "persist")
    else:
      return "/persist/"

  @staticmethod
  def config_root() -> str:
    if PC:
      return Paths.comma_home()
    else:
      return "/tmp/.comma"

  @staticmethod
  def shm_path() -> str:
    if PC and platform.system() == "Darwin":
      return "/tmp"  # This is not really shared memory on macOS, but it's the closest we can get
    if sys.platform == "win32":
      return TMP_DIR  # msgq reads %TEMP% for the same directory
    return "/dev/shm"
