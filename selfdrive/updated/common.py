import abc
import os

from pathlib import Path
import subprocess
from typing import List

from markdown_it import MarkdownIt

from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog


LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")
OVERLAY_UPPER = os.path.join(STAGING_ROOT, "upper")
OVERLAY_METADATA = os.path.join(STAGING_ROOT, "metadata")
OVERLAY_MERGED = os.path.join(STAGING_ROOT, "merged")
FINALIZED = os.path.join(STAGING_ROOT, "finalized")

OVERLAY_INIT = Path(os.path.join(BASEDIR, ".overlay_init"))


def run(cmd: list[str], cwd: str = None) -> str:
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')


class UpdateStrategy(abc.ABC):

  @abc.abstractmethod
  def get_available_channels(self) -> List[str]:
    """List of available channels to install, (branches, releases, etc)"""
    pass

  @abc.abstractmethod
  def get_current_channel(self) -> str:
    """Current channel installed"""
    pass

  @abc.abstractmethod
  def set_channel(self, channel: str) -> None:
    """Set the desired channel to install"""
    pass

  @abc.abstractmethod
  def update_ready(self) -> bool:
    """Check if an update is ready to be installed"""
    pass

  @abc.abstractmethod
  def update_avaiable(self) -> bool:
    """Check if an update is available for the current channel"""
    pass

  def describe_current_channel(self) -> tuple[str, str]:
    """Describe the current channel installed, (description, release_notes)"""
    pass

  def describe_ready_channel(self) -> tuple[str, str]:
    """Describe the channel that is ready to be installed, (description, release_notes)"""
    pass


def set_consistent_flag(consistent: bool) -> None:
  os.sync()
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  if consistent:
    consistent_file.touch()
  elif not consistent:
    consistent_file.unlink(missing_ok=True)
  os.sync()


def parse_release_notes(releases_md: str) -> bytes:
  try:
    r = releases_md.split(b'\n\n', 1)[0]  # Slice latest release notes
    try:
      return bytes(MarkdownIt().render(r.decode("utf-8")), encoding="utf-8")
    except Exception:
      return r + b"\n"
  except FileNotFoundError:
    pass
  except Exception:
    cloudlog.exception("failed to parse release notes")
  return b""
