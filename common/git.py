import subprocess
from openpilot.common.utils import cache
from openpilot.common.run import run_cmd, run_cmd_default


@cache
def get_commit(branch: str = "HEAD") -> str:
  return run_cmd_default(["git", "rev-parse", branch])


@cache
def get_commit_date(commit: str = "HEAD") -> str:
  return run_cmd_default(["git", "show", "--no-patch", "--format='%ct %ci'", commit])


@cache
def get_short_branch() -> str:
  return run_cmd_default(["git", "rev-parse", "--abbrev-ref", "HEAD"])


@cache
def get_branch() -> str:
  return run_cmd_default(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])


@cache
def get_origin() -> str:
  try:
    local_branch = run_cmd(["git", "name-rev", "--name-only", "HEAD"])
    tracking_remote = run_cmd(["git", "config", "branch." + local_branch + ".remote"])
    return run_cmd(["git", "config", "remote." + tracking_remote + ".url"])
  except subprocess.CalledProcessError:  # Not on a branch, fallback
    return run_cmd_default(["git", "config", "--get", "remote.origin.url"])


@cache
def get_normalized_origin() -> str:
  return get_origin() \
    .replace("git@", "", 1) \
    .replace(".git", "", 1) \
    .replace("https://", "", 1) \
    .replace(":", "/", 1)
