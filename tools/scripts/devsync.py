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

# fast in-process filter: drop events under these dir names without bothering git
IGNORED_DIR_PARTS = {
  ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
  ".venv", "venv", "node_modules", "build", ".scons_cache",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".o", ".os", ".so", ".d", ".swp", ".swo"}


class Debouncer:
  def __init__(self, delay: float, fn):
    self.delay = delay
    self.fn = fn
    self.timer: threading.Timer | None = None
    self.lock = threading.Lock()
    self.pending = 0

  def trigger(self):
    with self.lock:
      self.pending += 1
      if self.timer is not None:
        self.timer.cancel()
      self.timer = threading.Timer(self.delay, self._fire)
      self.timer.daemon = True
      self.timer.start()

  def _fire(self):
    with self.lock:
      n = self.pending
      self.pending = 0
      self.timer = None
    self.fn(n)


class Handler(FileSystemEventHandler):
  def __init__(self, debouncer: Debouncer):
    self.debouncer = debouncer

  @staticmethod
  def interesting(path: str) -> bool:
    p = Path(path)
    if any(part in IGNORED_DIR_PARTS for part in p.parts):
      return False
    if p.suffix in IGNORED_SUFFIXES:
      return False
    return True

  def on_any_event(self, event):
    if event.is_directory:
      return
    if not self.interesting(event.src_path):
      return
    self.debouncer.trigger()


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
  if args.delete:
    # only delete on the device if a tracked file was deleted locally
    cmd.append("--delete-missing-args")
    cmd.append("--ignore-missing-args")
  if args.dry_run:
    cmd.append("-n")
  src = str(args.src) + "/"
  dst = f"comma@{args.ip}:{args.remote}/"
  cmd += [src, dst]
  return cmd


def git_tracked_files(src: Path) -> bytes:
  return subprocess.check_output(
    ["git", "-C", str(src), "ls-files", "--recurse-submodules", "-z"]
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("ip", help="device IP / hostname")
  p.add_argument("--remote", default="/data/openpilot",
                 help="remote path on device")
  p.add_argument("--src", type=Path, default=BASEDIR,
                 help="local source directory")
  p.add_argument("-i", "--identity", default=None, help="ssh identity file")
  p.add_argument("--debounce", type=float, default=1.0,
                 help="seconds to coalesce events before syncing")
  p.add_argument("--delete", action="store_true",
                 help="propagate deletions of tracked files")
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--no-initial", action="store_true",
                 help="skip the full sync on startup")
  args = p.parse_args()

  print(f"[devsync] watching {args.src}")
  print(f"[devsync] target   comma@{args.ip}:{args.remote}")
  print(f"[devsync] debounce {args.debounce}s, delete={args.delete}, dry_run={args.dry_run}")

  syncing = threading.Lock()
  pending_again = threading.Event()

  def run_sync(events: int, initial: bool = False):
    if not syncing.acquire(blocking=False):
      # a sync is already in flight; ask it to run again when done
      pending_again.set()
      return
    try:
      while True:
        try:
          file_list = git_tracked_files(args.src)
        except subprocess.CalledProcessError as e:
          print(f"[devsync] git ls-files failed: {e}", file=sys.stderr)
          break
        n_listed = file_list.count(b"\0")

        cmd = build_rsync_cmd(args, initial=initial)
        t0 = time.monotonic()
        try:
          if initial:
            # stream rsync progress directly to the terminal
            rc = subprocess.run(cmd, input=file_list).returncode
            files: list[str] = []
            stderr = ""
          else:
            r = subprocess.run(cmd, input=file_list, capture_output=True)
            files = [ln for ln in r.stdout.decode().splitlines() if ln.strip()]
            rc = r.returncode
            stderr = r.stderr.decode()
        except FileNotFoundError:
          print("[devsync] rsync not found in PATH", file=sys.stderr)
          sys.exit(1)
        dt = time.monotonic() - t0

        if rc == 0:
          if initial:
            print(f"[devsync] initial sync done in {dt:.2f}s ({n_listed} tracked files)")
          else:
            ev = f" ({events} ev)" if events else ""
            if not files:
              print(f"[devsync] {dt:.2f}s{ev} · no changes")
            elif len(files) == 1:
              print(f"[devsync] {dt:.2f}s{ev} · {files[0]}")
            else:
              print(f"[devsync] {dt:.2f}s{ev} · {len(files)} files: {', '.join(files)}")
        else:
          print(f"[devsync] ERR rc={rc} in {dt:.2f}s")
          if stderr.strip():
            print(stderr.strip(), file=sys.stderr)

        if not pending_again.is_set():
          break
        pending_again.clear()
        events = 0
        initial = False
    finally:
      syncing.release()

  if not args.no_initial:
    print("[devsync] initial sync...")
    run_sync(0, initial=True)

  deb = Debouncer(args.debounce, run_sync)
  obs = Observer()
  obs.schedule(Handler(deb), str(args.src), recursive=True)
  obs.start()
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    print("\n[devsync] stopping")
  finally:
    obs.stop()
    obs.join()


if __name__ == "__main__":
  main()
