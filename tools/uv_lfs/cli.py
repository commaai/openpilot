"""CLI entry point for uv-lfs."""

import argparse
import os
import subprocess
import sys

from openpilot.tools.uv_lfs.config import git_dir, repo_root


def cmd_install(_args) -> int:
  """Configure git to use uv-lfs as the filter process."""
  gd = git_dir()

  # set filter.lfs.process for long-running filter protocol
  subprocess.check_call(["git", "config", "filter.lfs.process", "uv-lfs filter-process"])
  subprocess.check_call(["git", "config", "filter.lfs.clean", "uv-lfs clean %f"])
  subprocess.check_call(["git", "config", "filter.lfs.smudge", "uv-lfs smudge %f"])
  subprocess.check_call(["git", "config", "filter.lfs.required", "true"])

  # remove old git-lfs hooks if they exist
  hooks_dir = os.path.join(gd, "hooks")
  for hook in ("pre-push", "post-checkout", "post-commit", "post-merge"):
    hook_path = os.path.join(hooks_dir, hook)
    if os.path.isfile(hook_path):
      try:
        with open(hook_path) as f:
          content = f.read()
        if "git lfs" in content or "git-lfs" in content:
          os.unlink(hook_path)
      except OSError:
        pass

  print("uv-lfs: git filter configured")
  return 0


def cmd_checkout(_args) -> int:
  """Download and replace pointer files with real content."""
  from openpilot.tools.uv_lfs.batch_api import download_objects
  from openpilot.tools.uv_lfs.storage import has_object, read_object

  root = repo_root()
  pointers = _find_pointer_files(root)

  if not pointers:
    print("uv-lfs: no pointer files found")
    return 0

  # collect what we need to download
  to_download = []
  for _path, oid, size in pointers:
    if not has_object(oid):
      to_download.append((oid, size))

  if to_download:
    print(f"uv-lfs: downloading {len(to_download)} objects...")
    download_objects(to_download)

  # replace pointer files with real content
  replaced = 0
  for path, oid, _size in pointers:
    if has_object(oid):
      full_path = os.path.join(root, path)
      data = read_object(oid)
      with open(full_path, "wb") as f:
        f.write(data)
      replaced += 1

  print(f"uv-lfs: checkout complete, {replaced}/{len(pointers)} files")
  return 0


def cmd_push(_args) -> int:
  """Push locally cached LFS objects to the server."""
  from openpilot.tools.uv_lfs.batch_api import upload_objects
  from openpilot.tools.uv_lfs.storage import has_object

  root = repo_root()
  blobs = _find_lfs_blobs(root)

  oids_sizes = [(oid, size) for _, oid, size in blobs if has_object(oid)]
  if not oids_sizes:
    print("uv-lfs: nothing to push")
    return 0

  print(f"uv-lfs: pushing {len(oids_sizes)} objects...")
  uploaded = upload_objects(oids_sizes)
  print(f"uv-lfs: pushed {uploaded} objects")
  return 0


def cmd_prune(_args) -> int:
  """Remove cached objects not referenced by current HEAD."""
  from openpilot.tools.uv_lfs.storage import cached_oids, remove_object

  root = repo_root()
  blobs = _find_lfs_blobs(root)
  referenced = {oid for _, oid, _ in blobs}
  cached = cached_oids()

  to_remove = cached - referenced
  for oid in to_remove:
    remove_object(oid)

  print(f"uv-lfs: pruned {len(to_remove)} objects ({len(cached) - len(to_remove)} retained)")
  return 0


def cmd_ls_files(_args) -> int:
  """List LFS-tracked files."""
  from openpilot.tools.uv_lfs.storage import has_object

  root = repo_root()
  blobs = _find_lfs_blobs(root)

  for path, oid, _size in sorted(blobs):
    cached = "*" if has_object(oid) else "-"
    print(f"{oid[:12]} {cached} {path}")
  return 0


def cmd_filter_process(_args) -> int:
  """Run the long-running git filter protocol."""
  from openpilot.tools.uv_lfs.protocol import run_filter_process
  run_filter_process()
  return 0


def cmd_clean(_args) -> int:
  """Single-file clean filter (working tree → pointer)."""
  import hashlib

  from openpilot.tools.uv_lfs.pointer import format_pointer, parse_pointer
  from openpilot.tools.uv_lfs.storage import has_object, store_object

  data = sys.stdin.buffer.read()
  parsed = parse_pointer(data)
  if parsed is not None:
    sys.stdout.buffer.write(data)
    return 0

  oid = hashlib.sha256(data).hexdigest()
  size = len(data)
  if not has_object(oid):
    store_object(oid, data)
  sys.stdout.buffer.write(format_pointer(oid, size))
  return 0


def cmd_smudge(_args) -> int:
  """Single-file smudge filter (pointer → real content)."""
  from openpilot.tools.uv_lfs.batch_api import download_objects
  from openpilot.tools.uv_lfs.pointer import parse_pointer
  from openpilot.tools.uv_lfs.storage import has_object, read_object

  data = sys.stdin.buffer.read()
  parsed = parse_pointer(data)
  if parsed is None:
    sys.stdout.buffer.write(data)
    return 0

  oid, size = parsed
  if not has_object(oid):
    download_objects([(oid, size)], progress=False)

  if has_object(oid):
    sys.stdout.buffer.write(read_object(oid))
  else:
    sys.stdout.buffer.write(data)
  return 0


def _lfs_tracked_files(root: str) -> list[str]:
  """Return paths of all LFS-tracked files via .gitattributes patterns."""
  patterns = _lfs_patterns(root)
  if not patterns:
    return []

  cmd = ["git", "ls-files", "-z", "--"] + patterns
  try:
    output = subprocess.check_output(cmd, cwd=root)
  except subprocess.CalledProcessError:
    return []

  return [f for f in output.decode().split("\0") if f]


def _find_pointer_files(root: str) -> list[tuple[str, str, int]]:
  """Find LFS pointer files in the working tree (on-disk pointers only)."""
  from openpilot.tools.uv_lfs.pointer import parse_pointer

  pointers = []
  for path in _lfs_tracked_files(root):
    full = os.path.join(root, path)
    try:
      with open(full, "rb") as f:
        data = f.read(200)  # pointers are small
      parsed = parse_pointer(data)
      if parsed:
        pointers.append((path, parsed[0], parsed[1]))
    except (OSError, IsADirectoryError):
      continue

  return pointers


def _find_lfs_blobs(root: str) -> list[tuple[str, str, int]]:
  """Read LFS pointers from git HEAD blobs (works even when files are hydrated)."""
  from openpilot.tools.uv_lfs.pointer import parse_pointer

  blobs = []
  for path in _lfs_tracked_files(root):
    try:
      data = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=root)
      parsed = parse_pointer(data)
      if parsed:
        blobs.append((path, parsed[0], parsed[1]))
    except subprocess.CalledProcessError:
      continue

  return blobs


def _lfs_patterns(root: str) -> list[str]:
  """Extract file patterns with filter=lfs from .gitattributes."""
  patterns = []
  attrs_path = os.path.join(root, ".gitattributes")
  if not os.path.isfile(attrs_path):
    return patterns

  with open(attrs_path) as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      if "filter=lfs" in line:
        pattern = line.split()[0]
        patterns.append(pattern)

  return patterns


def main():
  parser = argparse.ArgumentParser(prog="uv-lfs", description="Git LFS replacement for openpilot")
  sub = parser.add_subparsers(dest="command")

  sub.add_parser("install", help="Configure git to use uv-lfs")
  sub.add_parser("checkout", help="Download and hydrate LFS files")
  sub.add_parser("push", help="Push LFS objects to server")
  sub.add_parser("prune", help="Remove unreferenced cached objects")
  sub.add_parser("ls-files", help="List LFS-tracked files")
  sub.add_parser("filter-process", help="Long-running filter protocol")
  p_clean = sub.add_parser("clean", help="Clean filter (stdin/stdout)")
  p_clean.add_argument("file", nargs="?")
  p_smudge = sub.add_parser("smudge", help="Smudge filter (stdin/stdout)")
  p_smudge.add_argument("file", nargs="?")

  args = parser.parse_args()

  commands = {
    "install": cmd_install,
    "checkout": cmd_checkout,
    "push": cmd_push,
    "prune": cmd_prune,
    "ls-files": cmd_ls_files,
    "filter-process": cmd_filter_process,
    "clean": cmd_clean,
    "smudge": cmd_smudge,
  }

  if args.command in commands:
    sys.exit(commands[args.command](args))
  else:
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
  main()
