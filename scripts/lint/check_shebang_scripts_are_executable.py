#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys


def staged_modes(filenames: list[str]) -> list[tuple[str, str]]:
  if not filenames:
    return []

  result = subprocess.run(
    ("git", "ls-files", "-z", "--stage", "--", *filenames),
    check=True,
    capture_output=True,
    text=True,
  )
  entries = result.stdout.rstrip("\0").split("\0") if result.stdout else []
  return [(entry.split(" ", 1)[0], entry.split("\t", 1)[1]) for entry in entries]


def has_shebang(filename: str) -> bool:
  with open(filename, "rb") as f:
    return f.read(2) == b"#!"


def check_shebang_scripts_are_executable(filenames: list[str]) -> int:
  failed = False
  for mode, filename in staged_modes(filenames):
    if mode != "100755" and has_shebang(filename):
      quoted = shlex.quote(filename)
      print("\n".join((
        f"{filename}: has a shebang but is not marked executable!",
        f"  If it is supposed to be executable, try: `chmod +x {quoted}`",
        "  If it is not supposed to be executable, double-check its shebang is wanted.\n",
      )), file=sys.stderr)
      failed = True

  return int(failed)


def main() -> int:
  parser = argparse.ArgumentParser(description="Check that tracked files with shebangs are executable.")
  parser.add_argument("filenames", nargs="*")
  args = parser.parse_args()
  return check_shebang_scripts_are_executable(args.filenames)


if __name__ == "__main__":
  raise SystemExit(main())
