#!/usr/bin/env python3

import importlib
import sys
import zipapp
from argparse import ArgumentParser

from openpilot.common.basedir import BASEDIR

ENTRYPOINT = 'main'

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

  module_entrypoint = f'{args.module}:{ENTRYPOINT}'

  zipapp.create_archive(BASEDIR + '/openpilot', target=args.output, interpreter='/usr/bin/env python3', main=module_entrypoint)
