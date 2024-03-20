import subprocess


def run_cmd(cmd: list[str]) -> str:
  return subprocess.check_output(cmd, encoding='utf8').strip()


def run_cmd_default(cmd: list[str], default: str = "") -> str:
  try:
    return run_cmd(cmd)
  except subprocess.CalledProcessError:
    return default

