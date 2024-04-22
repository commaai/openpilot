import subprocess
from openpilot.common.utils import cache
from openpilot.common.run import run_cmd, run_cmd_default


@cache
def get_commit(cwd: str = None, branch: str = "HEAD") -> str:
  return run_cmd_default(["git", "rev-parse", branch], cwd=cwd)


@cache
def get_commit_date(cwd: str = None, commit: str = "HEAD") -> str:
  return run_cmd_default(["git", "show", "--no-patch", "--format='%ct %ci'", commit], cwd=cwd)


@cache
def get_short_branch(cwd: str = None) -> str:
  return run_cmd_default(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)


@cache
def get_branch(cwd: str = None) -> str:
  return run_cmd_default(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=cwd)


@cache
def get_origin(cwd: str = None) -> str:
  try:
    local_branch = run_cmd(["git", "name-rev", "--name-only", "HEAD"], cwd=cwd)
    tracking_remote = run_cmd(["git", "config", "branch." + local_branch + ".remote"], cwd=cwd)
    return run_cmd(["git", "config", "remote." + tracking_remote + ".url"], cwd=cwd)
  except subprocess.CalledProcessError:  # Not on a branch, fallback
    return run_cmd_default(["git", "config", "--get", "remote.origin.url"], cwd=cwd)


@cache
def get_normalized_origin(cwd: str = None) -> str:
  return get_origin(cwd) \
    .replace("git@", "", 1) \
    .replace(".git", "", 1) \
    .replace("https://", "", 1) \
    .replace(":", "/", 1)
