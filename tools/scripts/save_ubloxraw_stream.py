#!/usr/bin/env python
import argparse
import os
import sys
from common.basedir import BASEDIR
from tools.lib.logreader import MultiLogIterator
from tools.lib.route import Route

os.environ['BASEDIR'] = BASEDIR


def get_arg_parser():
  parser = argparse.ArgumentParser(
      description="Unlogging and save to file",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("data_dir", nargs='?',
                              help="Path to directory in which log and camera files are located.")
  parser.add_argument("route_name", type=(lambda x: x.replace("#", "|")), nargs="?",
                      help="The route whose messages will be published.")
  parser.add_argument("--out_path", nargs='?', default='/data/ubloxRaw.stream',
                      help="Output pickle file path")
  return parser


def main(argv):
  args = get_arg_parser().parse_args(sys.argv[1:])
  if not args.data_dir:
    print('Data directory invalid.')
    return

  if not args.route_name:
    # Extract route name from path
    args.route_name = os.path.basename(args.data_dir)
    args.data_dir = os.path.dirname(args.data_dir)

  route = Route(args.route_name, args.data_dir)
  lr = MultiLogIterator(route.log_paths(), wraparound=False)

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
  sys.exit(main(sys.argv[1:]))
