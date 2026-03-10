#!/usr/bin/env python3

import os
import importlib
import shutil
import subprocess
import sys
import tempfile
import zipapp
from argparse import ArgumentParser
from pathlib import Path

from openpilot.common.basedir import BASEDIR

# TODO: change to prefix -> folders
# DIRS = {
#   'openpilot': ['common', 'selfdrive', 'system', 'third_party', 'tools'],  # these are symlinks
#   'cereal': [],
# }

DIRS = {
  # 'openpilot': ['common', 'selfdrive', 'system', 'third_party', 'tools'],  # these are symlinks  TODO: ls /openpilot
  'openpilot': os.listdir(os.path.join(BASEDIR, 'openpilot')),  # these are symlinks  TODO: ls /openpilot
  '': ['cereal'],
}

MODULES = ['cereal']
# SYMLINKS = os
EXTS = ['.png', '.py', '.ttf', '.capnp', '.json', '.fnt', '.mo', '.po']
INTERPRETER = '/usr/bin/env python3'


def get_tracked_files():
  result = subprocess.run(['git', 'ls-files'], cwd=BASEDIR, capture_output=True, text=True, check=True)
  return set(result.stdout.splitlines())


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

  tracked_files = get_tracked_files()

  # print(tracked_files)

  with tempfile.TemporaryDirectory() as tmp:
    for prefix, folders in DIRS.items():
      for folder in folders:
        for root, _, files in os.walk(os.path.join(BASEDIR, folder)):
          # print(root, files)
          for file in files:
            path = os.path.join(root, file).replace(BASEDIR, '').removeprefix('/')
            if path in tracked_files:
              dest = os.path.join(tmp, prefix, path)
              print('COPYING!!!', path, dest)
              os.makedirs(os.path.dirname(dest), exist_ok=True)
              copy(os.path.join(BASEDIR, path), dest)

    # --- START WORKS
    # for directory in DIRS:
    #   for root, _, files in os.walk(os.path.join(BASEDIR, directory), followlinks=True):
    #     # if 'selfdrive/ui' not in root:
    #     #   continue
    #     # print(root)
    #     for file in files:
    #       path = os.path.join(root, file).replace(BASEDIR, '').removeprefix('/openpilot/')
    #       # print('path', path)
    #       if path in tracked_files:
    #         print('COPYING!!!', path)
    #         dest = os.path.join(tmp, path)
    #         os.makedirs(os.path.dirname(dest), exist_ok=True)
    #         # shutil.copy2(os.path.join(BASEDIR, path), dest, follow_symlinks=True)
    #         copy(os.path.join(BASEDIR, path), dest)
    #       # print((root, files))
    # --- END WORKS

    # for file in get_tracked_files():
    #   print(file)

    # for directory in DIRS:
    #   shutil.copytree(BASEDIR + '/' + directory, tmp + '/' + directory, symlinks=False, dirs_exist_ok=True, copy_function=copy)
    entry = f'{args.module}:{args.entrypoint}'
    zipapp.create_archive(tmp, target=args.output, interpreter=INTERPRETER, main=entry)

  print(f'created executable {Path(args.output).resolve()}')
