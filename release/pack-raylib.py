#!/usr/bin/env python3

import importlib
import shutil
import sys
import tempfile
import zipapp
from argparse import ArgumentParser
from pathlib import Path

from openpilot.common.basedir import BASEDIR


ENTRYPOINT = 'main'
INTERPRETER = '/usr/bin/env python3'
EXTS = ['.py', '.png']


def copy(src, dest, follow_symlinks=False):
  if any(src.endswith(ext) for ext in EXTS):
    shutil.copy2(src, dest, follow_symlinks=follow_symlinks)


if __name__ == '__main__':
  parser = ArgumentParser(prog='pack-raylib.py', description='Package a raylib UI into a portable executable.', epilog='comma.ai')
  parser.add_argument('-o', '--output', help='output file')
  parser.add_argument('module')
  args = parser.parse_args()

  if not args.output:
    args.output = args.module

  try:
    mod = importlib.import_module(args.module)
  except ModuleNotFoundError:
    print(f'{args.module} not found, typo?')
    sys.exit(1)

  if not hasattr(mod, ENTRYPOINT):
    print(f'{args.module} does not have a {ENTRYPOINT}() function')
    sys.exit(1)

  with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(BASEDIR + '/openpilot', tmp, symlinks=False, dirs_exist_ok=True, copy_function=copy)
    entry = f'{args.module}:{ENTRYPOINT}'
    zipapp.create_archive(tmp, target=args.output, interpreter=INTERPRETER, main=entry)

  print(f'created executable {Path(args.output).resolve()}')

