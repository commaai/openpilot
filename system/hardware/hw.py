import os
from selfdrive.hardware import PC
from pathlib import Path

class Paths:
  @staticmethod
  def comma_home() -> str:
      return os.path.join(str(Path.home()), ".comma")

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
      return os.path.join(Paths.comma_home(), "log" + os.environ.get("OPENPILOT_PREFIX", ""))
    else:
      return "/data/log/"