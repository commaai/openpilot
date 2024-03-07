import abc
import os

from pathlib import Path
import subprocess
from typing import List

from markdown_it import MarkdownIt
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


LOCK_FILE = os.getenv("UPDATER_LOCK_FILE", "/tmp/safe_staging_overlay.lock")
STAGING_ROOT = os.getenv("UPDATER_STAGING_ROOT", "/data/safe_staging")
FINALIZED = os.path.join(STAGING_ROOT, "finalized")


def run(cmd: list[str], cwd: str = None) -> str:
  return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, encoding='utf8')


class UpdateStrategy(abc.ABC):
  def __init__(self):
    self.params = Params()

  @abc.abstractmethod
  def init(self) -> None:
    pass

  @abc.abstractmethod
  def cleanup(self) -> None:
    pass

  @abc.abstractmethod
  def get_available_channels(self) -> List[str]:
    """List of available channels to install, (branches, releases, etc)"""

  @abc.abstractmethod
  def current_channel(self) -> str:
    """Current channel installed"""

  @abc.abstractmethod
  def fetched_path(self) -> str:
    """Path to the fetched update"""

  @property
  def target_channel(self) -> str:
    """Target Channel"""
    b: str | None = self.params.get("UpdaterTargetBranch", encoding='utf-8')
    if b is None:
      b = self.current_channel()
    return b

  @abc.abstractmethod
  def update_ready(self) -> bool:
    """Check if an update is ready to be installed"""

  @abc.abstractmethod
  def update_available(self) -> bool:
    """Check if an update is available for the current channel"""

  @abc.abstractmethod
  def describe_current_channel(self) -> tuple[str, str]:
    """Describe the current channel installed, (description, release_notes)"""

  @abc.abstractmethod
  def describe_ready_channel(self) -> tuple[str, str]:
    """Describe the channel that is ready to be installed, (description, release_notes)"""

  @abc.abstractmethod
  def fetch_update(self) -> None:
    pass

  @abc.abstractmethod
  def finalize_update(self) -> None:
    pass


def set_consistent_flag(consistent: bool) -> None:
  os.sync()
  consistent_file = Path(os.path.join(FINALIZED, ".overlay_consistent"))
  if consistent:
    consistent_file.touch()
  elif not consistent:
    consistent_file.unlink(missing_ok=True)
  os.sync()


def parse_release_notes(releases_md: str) -> str:
  try:
    r = releases_md.split('\n\n', 1)[0]  # Slice latest release notes
    try:
      return str(MarkdownIt().render(r))
    except Exception:
      return r + "\n"
  except FileNotFoundError:
    pass
  except Exception:
    cloudlog.exception("failed to parse release notes")
  return ""
