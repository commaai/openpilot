#!/usr/bin/env python3

import zipapp
from argparse import ArgumentParser

from openpilot.common.basedir import BASEDIR

if __name__ == '__main__':
  parser = ArgumentParser(prog='pack-raylib.py', description='Package a raylib UI into a portable executable.', epilog='comma.ai')
  parser.add_argument('module')
  parser.add_argument('-o', '--output', help='output file')
  args = parser.parse_args()
  zipapp.create_archive(BASEDIR + '/openpilot', target=args.output, interpreter='/usr/bin/env python3', main=args.module)
