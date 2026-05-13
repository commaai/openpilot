#!/usr/bin/env python3
import argparse
import os
import sys
from openpilot.common.basedir import BASEDIR
from openpilot.tools.lib.logreader import LogReader

os.environ['BASEDIR'] = BASEDIR


def get_arg_parser():
  parser = argparse.ArgumentParser(
      description="Unlogging and save to file",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("route", type=(lambda x: x.replace("#", "|")), nargs="?",
                      help="The route whose messages will be published.")
  parser.add_argument("--out_path", nargs='?', default='/data/ubloxRaw.stream',
                      help="Output pickle file path")
  return parser


def main():
  args = get_arg_parser().parse_args(sys.argv[1:])

  lr = LogReader(args.route)

  with open(args.out_path, 'wb') as f:
    try:
      done = False
      i = 0
      while not done:
        msg = next(lr)
        if not msg:
          break
        smsg = msg.as_builder()
        typ = smsg.which()
        if typ == 'ubloxRaw':
          f.write(smsg.to_bytes())
          i += 1
    except StopIteration:
      print('All done')
  print(f'Writed {i} msgs')


if __name__ == "__main__":
  main()
