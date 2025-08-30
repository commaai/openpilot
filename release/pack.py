#!/usr/bin/env python3

import importlib
import shutil
import sys
import tempfile
import zipapp
from argparse import ArgumentParser
from pathlib import Path

from openpilot.common.basedir import BASEDIR


DIRS = ['cereal', 'openpilot']
EXTS = ['.png', '.py', '.ttf', '.capnp']
INTERPRETER = '/usr/bin/env python3'


def copy(src, dest):
  if any(src.endswith(ext) for ext in EXTS):
    shutil.copy2(src, dest, follow_symlinks=True)


if __name__ == '__main__':
  parser = ArgumentParser(prog='pack.py', description="package script into a portable executable", epilog='comma.ai')
  parser.add_argument('-e', '--entrypoint', help="function to call in module, default is 'main'", default='main')
  parser.add_argument('-o', '--output', help='output file')
  parser.add_argument('module', help="the module to target, e.g. 'openpilot.system.ui.spinner'")
  args = parser.parse_args()

  if not args.output:
    args.output = args.module

  try:
    mod = importlib.import_module(args.module)
  except ModuleNotFoundError:
    print(f'{args.module} not found, typo?')
    sys.exit(1)

  if not hasattr(mod, args.entrypoint):
    print(f'{args.module} does not have a {args.entrypoint}() function, typo?')
    sys.exit(1)

  with tempfile.TemporaryDirectory() as tmp:
    for directory in DIRS:
      shutil.copytree(BASEDIR + '/' + directory, tmp + '/' + directory, symlinks=False, dirs_exist_ok=True, copy_function=copy)
    entry = f'{args.module}:{args.entrypoint}'
    zipapp.create_archive(tmp, target=args.output, interpreter=INTERPRETER, main=entry)

  print(f'created executable {Path(args.output).resolve()}')
