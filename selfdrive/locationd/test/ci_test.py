#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse
import tempfile

from selfdrive.locationd.test.ubloxd_py_test import parser_test
from selfdrive.locationd.test.ubloxd_regression_test import compare_results


def mkdirs_exists_ok(path):
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise


def main(args):
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  ubloxd_dir = os.path.join(cur_dir, '../')

  cc_output_dir = os.path.join(args.output_dir, 'cc')
  mkdirs_exists_ok(cc_output_dir)

  py_output_dir = os.path.join(args.output_dir, 'py')
  mkdirs_exists_ok(py_output_dir)

  archive_file = os.path.join(cur_dir, args.stream_gz_file)

  try:
    print('Extracting stream file')
    subprocess.check_call(['tar', 'zxf', archive_file], cwd=tempfile.gettempdir())
    stream_file_path = os.path.join(tempfile.gettempdir(), 'ubloxRaw.stream')

    if not os.path.isfile(stream_file_path):
      print('Extract file failed')
      sys.exit(-3)

    print('Compiling test app...')
    subprocess.check_call(["make", "ubloxd_test"], cwd=ubloxd_dir)

    print('Run regression test - CC parser...')
    if args.valgrind:
      subprocess.check_call(["valgrind", "--leak-check=full", os.path.join(ubloxd_dir, 'ubloxd_test'), stream_file_path, cc_output_dir])
    else:
      subprocess.check_call([os.path.join(ubloxd_dir, 'ubloxd_test'), stream_file_path, cc_output_dir])

    print('Running regression test - py parser...')
    parser_test(stream_file_path, py_output_dir)

    print('Running regression test - compare result...')
    r = compare_results(cc_output_dir, py_output_dir)

    print('All done!')

    subprocess.check_call(["rm", stream_file_path])
    subprocess.check_call(["rm", '-rf', cc_output_dir])
    subprocess.check_call(["rm", '-rf', py_output_dir])
    sys.exit(r)

  except subprocess.CalledProcessError as e:
    print('CI test failed with {}'.format(e.returncode))
    sys.exit(e.returncode)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Ubloxd CI test",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("stream_gz_file", nargs='?', default='ubloxRaw.tar.gz',
                      help="UbloxRaw data stream zip file")

  parser.add_argument("output_dir", nargs='?', default='out',
                      help="Output events temp directory")

  parser.add_argument("--valgrind", default=False, action='store_true',
                      help="Run in valgrind")

  args = parser.parse_args()
  main(args)
