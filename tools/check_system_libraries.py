#!/usr/bin/env python3
import argparse
from collections.abc import Iterable
import os
from pathlib import Path
import re
import subprocess
import sys


RESOLVED_RE = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)")
LOADER_RE = re.compile(r"^\s*(/\S+)\s+\(")
SONAME_RE = re.compile(r"\(SONAME\)\s+Library soname: \[(.+)\]")


def elf_files(roots: Iterable[Path]) -> list[Path]:
  root_paths = [root.resolve() for root in roots]
  files = set()
  for root in root_paths:
    for current, dirs, names in os.walk(root):
      dirs[:] = [name for name in dirs if name != ".git"]
      for name in names:
        path = Path(current) / name
        try:
          resolved = path.resolve()
          if not any(resolved.is_relative_to(root_path) for root_path in root_paths):
            continue
          with path.open("rb") as f:
            if f.read(4) == b"\x7fELF":
              files.add(resolved)
        except (OSError, RuntimeError):
          continue
  return sorted(files)


def dynamic_dependencies(path: Path) -> list[tuple[str, Path | None]]:
  env = os.environ.copy()
  env.pop("LD_LIBRARY_PATH", None)
  env.pop("LD_PRELOAD", None)
  output = subprocess.run(["ldd", path], text=True, capture_output=True, env=env, check=False).stdout
  dependencies = []
  for line in output.splitlines():
    if match := RESOLVED_RE.match(line):
      name, resolved = match.groups()
      dependencies.append((name, None if resolved == "not" else Path(resolved).resolve()))
    elif match := LOADER_RE.match(line):
      resolved = Path(match.group(1)).resolve()
      dependencies.append((resolved.name, resolved))
  return dependencies


def soname(path: Path) -> str | None:
  output = subprocess.check_output(["readelf", "-dW", path], text=True, stderr=subprocess.DEVNULL)
  for line in output.splitlines():
    if match := SONAME_RE.search(line):
      return match.group(1)
  return None


def check_system_libraries(roots: Iterable[Path], allowed_system_libs: set[str]) -> None:
  root_paths = [root.resolve() for root in roots]
  binaries = elf_files(root_paths)
  vendored_libs = {binary.name for binary in binaries}
  vendored_libs.update(name for binary in binaries if (name := soname(binary)) is not None)

  violations = [
    (binary, library)
    for binary in binaries
    for library, resolved in dynamic_dependencies(binary)
    if not (resolved is not None and any(resolved.is_relative_to(root) for root in root_paths))
    and not (resolved is None and library in vendored_libs)
    and library not in allowed_system_libs
  ]
  if violations:
    details = "\n".join(f"  {binary}: {library}" for binary, library in violations)
    raise RuntimeError(f"unexpected system library dependencies:\n{details}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("roots", nargs="+", type=Path)
  parser.add_argument("--allow", action="append", default=[])
  args = parser.parse_args()
  try:
    check_system_libraries(args.roots, set(args.allow))
  except RuntimeError as e:
    print(e, file=sys.stderr)
    raise SystemExit(1) from None


if __name__ == "__main__":
  main()
