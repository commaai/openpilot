from functools import cache
import subprocess
from openpilot.common.safe import check_output
from openpilot.common.utils import run_cmd


def _git_output(cmd: list[str], cwd: str | None = None) -> str:
  return check_output(cmd, b"", cwd=cwd).decode().strip()


@cache
def get_commit(cwd: str | None = None, branch: str = "HEAD") -> str:
  return _git_output(["git", "rev-parse", branch], cwd)


@cache
def get_commit_date(cwd: str | None = None, commit: str = "HEAD") -> str:
  return _git_output(["git", "show", "--no-patch", "--format='%ct %ci'", commit], cwd)


@cache
def get_short_branch(cwd: str | None = None) -> str:
  return _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)


@cache
def get_branch(cwd: str | None = None) -> str:
  return _git_output(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd)


@cache
def get_origin(cwd: str | None = None) -> str:
  try:
    local_branch = run_cmd(["git", "name-rev", "--name-only", "HEAD"], cwd=cwd)
    tracking_remote = run_cmd(["git", "config", "branch." + local_branch + ".remote"], cwd=cwd)
    return run_cmd(["git", "config", "remote." + tracking_remote + ".url"], cwd=cwd)
  except subprocess.CalledProcessError:  # Not on a branch, fallback
    return _git_output(["git", "config", "--get", "remote.origin.url"], cwd)


@cache
def get_normalized_origin(cwd: str | None = None) -> str:
  return get_origin(cwd) \
    .replace("git@", "", 1) \
    .replace(".git", "", 1) \
    .replace("https://", "", 1) \
    .replace(":", "/", 1)
