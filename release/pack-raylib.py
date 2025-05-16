#!/usr/bin/env python3

import importlib
import os
import shutil
import sys
import zipapp
from argparse import ArgumentParser

from openpilot.common.basedir import BASEDIR


ENTRYPOINT = 'main'
INTERPRETER = '/usr/bin/env python3'


def copy_directory(src_dir, dst_dir):
    """
    Copy all files from source directory to destination directory, following symlinks.

    Args:
        src_dir (str): Path to source directory
        dst_dir (str): Path to destination directory
    """
    # Ensure source directory exists
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory '{src_dir}' does not exist")

    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Walk through source directory
    for root, _, files in os.walk(src_dir, followlinks=True):
        # Calculate relative path from source directory
        rel_path = os.path.relpath(root, src_dir)
        # Create corresponding directory in destination
        dst_root = os.path.join(dst_dir, rel_path)
        os.makedirs(dst_root, exist_ok=True)

        # Copy each file
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_root, file)
            # Copy file, preserving metadata and following symlinks
            shutil.copy2(src_file, dst_file, follow_symlinks=True)


if __name__ == '__main__':
  parser = ArgumentParser(prog='pack-raylib.py', description='Package a raylib UI into a portable executable.', epilog='comma.ai')
  parser.add_argument('module')
  parser.add_argument('-o', '--output', help='output file')
  args = parser.parse_args()

  try:
    mod = importlib.import_module(args.module)
  except ModuleNotFoundError:
    print(f'{args.module} not found, typo?')
    sys.exit(1)

  if not hasattr(mod, ENTRYPOINT):
    print(f'{args.module} does not have a {ENTRYPOINT}() function')
    sys.exit(1)

  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    copy_directory(BASEDIR + '/openpilot', tmp)
    entry = f'{args.module}:{ENTRYPOINT}'
    zipapp.create_archive(tmp, target=args.output, interpreter=INTERPRETER, main=entry)
