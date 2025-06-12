import subprocess


def run_cmd(cmd: list[str], cwd=None, env=None) -> str:
  return subprocess.check_output(cmd, encoding='utf8', cwd=cwd, env=env).strip()


def run_cmd_default(cmd: list[str], default: str = "", cwd=None, env=None) -> str:
  try:
    return run_cmd(cmd, cwd=cwd, env=env)
  except subprocess.CalledProcessError:
    return default

