import subprocess
from contextlib import contextmanager
from subprocess import Popen, PIPE, TimeoutExpired


def run_cmd(cmd: list[str], cwd=None, env=None) -> str:
  return subprocess.check_output(cmd, encoding='utf8', cwd=cwd, env=env).strip()


def run_cmd_default(cmd: list[str], default: str = "", cwd=None, env=None) -> str:
  try:
    return run_cmd(cmd, cwd=cwd, env=env)
  except subprocess.CalledProcessError:
    return default


@contextmanager
def managed_proc(cmd: list[str], env: dict[str, str]):
  proc = Popen(cmd, env=env, stdout=PIPE, stderr=PIPE)
  try:
    yield proc
  finally:
    if proc.poll() is None:
      proc.terminate()
    try:
      proc.wait(timeout=5)
    except TimeoutExpired:
      proc.kill()
