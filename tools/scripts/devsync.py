#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import threading
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from openpilot.common.basedir import BASEDIR


def build_rsync_cmd(args) -> list[str]:
  ssh = [
    "ssh",
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath=/tmp/devsync-{args.ip}.ctl",
    "-o", "ControlPersist=10m",
    "-o", "StrictHostKeyChecking=accept-new",
  ]
  if args.identity:
    ssh += ["-i", args.identity]

  return [
    "rsync", "-az",
    "--files-from=-", "--from0",
    "-e", " ".join(shlex.quote(p) for p in ssh),
    "--out-format=%n",
    BASEDIR + "/", f"comma@{args.ip}:{args.remote}/",
  ]


def git_tracked_files() -> bytes:
  return subprocess.check_output(
    ["git", "-C", BASEDIR, "ls-files", "--recurse-submodules", "-z"]
  )


class Handler(FileSystemEventHandler):
  def __init__(self, sync_fn):
    self.dirty = threading.Event()
    self.sync_fn = sync_fn

  def on_any_event(self, event):
    if not event.is_directory:
      self.dirty.set()

  def run(self):
    while True:
      time.sleep(1)
      if self.dirty.is_set():
        self.dirty.clear()
        self.sync_fn()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("ip", help="device IP / hostname")
  p.add_argument("--remote", default="/data/openpilot", help="remote path on device")
  p.add_argument("-i", "--identity", default=None, help="ssh identity file")
  args = p.parse_args()

  print(f"[devsync] watching {BASEDIR}")
  print(f"[devsync] target   comma@{args.ip}:{args.remote}")

  def run_sync():
    file_list = git_tracked_files()
    cmd = build_rsync_cmd(args)
    t0 = time.monotonic()
    r = subprocess.run(cmd, input=file_list, capture_output=True)
    dt = time.monotonic() - t0
    if r.returncode:
      print(f"[devsync] ERR rc={r.returncode} in {dt:.2f}s")
      return
    files = [ln for ln in r.stdout.decode().splitlines() if ln.strip()]
    msg = f"{len(files)} files: {', '.join(files)}" if files else "no changes"
    print(f"[devsync] {dt:.2f}s · {msg}")

  run_sync()

  handler = Handler(run_sync)
  obs = Observer()
  obs.schedule(handler, BASEDIR, recursive=True)
  obs.start()
  handler.run()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\n[devsync] stopping")
