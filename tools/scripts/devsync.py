#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from openpilot.common.basedir import BASEDIR


def build_ssh_cmd(args) -> str:
  ctl = f"/tmp/devsync-{args.ip}.ctl"
  parts = [
    "ssh",
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={ctl}",
    "-o", "ControlPersist=10m",
    "-o", "StrictHostKeyChecking=accept-new",
  ]
  if args.identity:
    parts += ["-i", args.identity]
  return " ".join(shlex.quote(p) for p in parts)


def build_rsync_cmd(args, initial: bool) -> list[str]:
  cmd = [
    "rsync", "-az",
    "--files-from=-", "--from0",
    "-e", build_ssh_cmd(args),
  ]
  if initial:
    cmd.append("--info=progress2,stats1")
  else:
    cmd.append("--out-format=%n")
  src = str(args.src) + "/"
  dst = f"comma@{args.ip}:{args.remote}/"
  cmd += [src, dst]
  return cmd


def git_tracked_files(src: Path) -> bytes:
  return subprocess.check_output(
    ["git", "-C", str(src), "ls-files", "--recurse-submodules", "-z"]
  )


class Handler(FileSystemEventHandler):
  def __init__(self):
    self.events = 0
    self.lock = threading.Lock()

  def on_any_event(self, event):
    if not event.is_directory:
      with self.lock:
        self.events += 1

  def drain(self) -> int:
    with self.lock:
      n, self.events = self.events, 0
    return n


def main():
  p = argparse.ArgumentParser()
  p.add_argument("ip", help="device IP / hostname")
  p.add_argument("--remote", default="/data/openpilot", help="remote path on device")
  p.add_argument("--src", type=Path, default=BASEDIR, help="local source directory")
  p.add_argument("-i", "--identity", default=None, help="ssh identity file")
  args = p.parse_args()

  print(f"[devsync] watching {args.src}")
  print(f"[devsync] target   comma@{args.ip}:{args.remote}")

  def run_sync(n_events: int, initial: bool = False):
    file_list = git_tracked_files(args.src)
    n_listed = file_list.count(b"\0")
    cmd = build_rsync_cmd(args, initial=initial)
    t0 = time.monotonic()
    if initial:
      # stream rsync progress directly to the terminal
      rc = subprocess.run(cmd, input=file_list).returncode
      files: list[str] = []
    else:
      r = subprocess.run(cmd, input=file_list, capture_output=True)
      files = [ln for ln in r.stdout.decode().splitlines() if ln.strip()]
      rc = r.returncode
      if rc and r.stderr.strip():
        print(r.stderr.decode().strip(), file=sys.stderr)
    dt = time.monotonic() - t0

    if rc != 0:
      print(f"[devsync] ERR rc={rc} in {dt:.2f}s")
    elif initial:
      print(f"[devsync] initial sync done in {dt:.2f}s ({n_listed} tracked files)")
    else:
      ev = f" ({n_events} ev)" if n_events else ""
      msg = f"{len(files)} files: {', '.join(files)}" if files else "no changes"
      print(f"[devsync] {dt:.2f}s{ev} · {msg}")

  print("[devsync] initial sync...")
  run_sync(0, initial=True)

  handler = Handler()
  obs = Observer()
  obs.schedule(handler, str(args.src), recursive=True)
  obs.start()
  try:
    while True:
      time.sleep(1)
      n = handler.drain()
      if n:
        run_sync(n)
  except KeyboardInterrupt:
    print("\n[devsync] stopping")
  finally:
    obs.stop()
    obs.join()


if __name__ == "__main__":
  main()
