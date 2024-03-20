import subprocess


def run_cmd(cmd: list[str], cwd=None) -> str:
  return subprocess.check_output(cmd, encoding='utf8', cwd=cwd).strip()


def run_cmd_default(cmd: list[str], default: str = "", cwd=None) -> str:
  try:
    return run_cmd(cmd, cwd=cwd)
  except subprocess.CalledProcessError:
    return default

