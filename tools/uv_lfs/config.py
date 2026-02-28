"""Read LFS configuration and resolve repo paths."""

import configparser
import subprocess
import sys


def git(*args: str) -> str:
  return subprocess.check_output(["git"] + list(args), text=True, stderr=subprocess.DEVNULL).strip()


def repo_root() -> str:
  try:
    return git("rev-parse", "--show-toplevel")
  except subprocess.CalledProcessError:
    print("uv-lfs: not a git repository", file=sys.stderr)
    sys.exit(128)


def git_dir() -> str:
  try:
    return git("rev-parse", "--absolute-git-dir")
  except subprocess.CalledProcessError:
    print("uv-lfs: not a git repository", file=sys.stderr)
    sys.exit(128)


def lfs_url() -> str:
  """Read the LFS server URL from .lfsconfig or git config."""
  root = repo_root()
  cfg = configparser.ConfigParser()
  cfg.read(f"{root}/.lfsconfig")
  url = cfg.get("lfs", "url", fallback=None)
  if url is None:
    url = git("config", "lfs.url")
  return url


def lfs_push_url() -> str:
  """Read the LFS push URL (SSH) from .lfsconfig or git config."""
  root = repo_root()
  cfg = configparser.ConfigParser()
  cfg.read(f"{root}/.lfsconfig")
  url = cfg.get("lfs", "pushurl", fallback=None)
  if url is None:
    try:
      url = git("config", "lfs.pushurl")
    except subprocess.CalledProcessError:
      url = lfs_url()
  return url
