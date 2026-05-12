#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
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
    args.src + "/", f"comma@{args.ip}:{args.remote}/",
  ]


def git_tracked_files(src: str) -> bytes:
  return subprocess.check_output(
    ["git", "-C", src, "ls-files", "--recurse-submodules", "-z"]
  )


class Handler(FileSystemEventHandler):
  def __init__(self, sync_fn):
    self.events = 0
    self.lock = threading.Lock()
    self.sync_fn = sync_fn

  def on_any_event(self, event):
    if not event.is_directory:
      with self.lock:
        self.events += 1

  def run(self):
    while True:
      time.sleep(1)
      with self.lock:
        n, self.events = self.events, 0
      if n:
        self.sync_fn(n)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("ip", help="device IP / hostname")
  p.add_argument("--remote", default="/data/openpilot", help="remote path on device")
  p.add_argument("--src", default=BASEDIR, help="local source directory")
  p.add_argument("-i", "--identity", default=None, help="ssh identity file")
  args = p.parse_args()

  print(f"[devsync] watching {args.src}")
  print(f"[devsync] target   comma@{args.ip}:{args.remote}")

  def run_sync(n_events: int = 0):
    file_list = git_tracked_files(args.src)
    cmd = build_rsync_cmd(args)
    t0 = time.monotonic()
    r = subprocess.run(cmd, input=file_list, capture_output=True)
    dt = time.monotonic() - t0
    if r.returncode:
      print(f"[devsync] ERR rc={r.returncode} in {dt:.2f}s")
      if r.stderr.strip():
        print(r.stderr.decode().strip(), file=sys.stderr)
      return
    files = [ln for ln in r.stdout.decode().splitlines() if ln.strip()]
    ev = f" ({n_events} ev)" if n_events else ""
    msg = f"{len(files)} files: {', '.join(files)}" if files else "no changes"
    print(f"[devsync] {dt:.2f}s{ev} · {msg}")

  run_sync()

  handler = Handler(run_sync)
  obs = Observer()
  obs.schedule(handler, args.src, recursive=True)
  obs.start()
  handler.run()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass
