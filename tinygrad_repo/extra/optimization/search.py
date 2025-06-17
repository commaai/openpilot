import argparse
from extra.optimization.helpers import ast_str_to_lin, time_linearizer

from tinygrad import dtypes
from tinygrad.helpers import BEAM, getenv
from tinygrad.device import Device, Compiled
from tinygrad.codegen.kernel import Kernel
from tinygrad.engine.search import beam_search, bufs_from_lin


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run a search for the optimal opts for a kernel", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--ast", type=str, default=None, help="the ast for the kernel to be optimized")
  parser.add_argument("--file", type=str, default=None, help="a file containing asts to be optimized, one per line")
  args = parser.parse_args()

  device: Compiled = Device[Device.DEFAULT]
  print(f"optimizing for {Device.DEFAULT}")

  if args.ast is not None:
    ast_strs = [args.ast]
  elif args.file is not None:
    with open(args.file, 'r') as file:
      ast_strs = file.readlines()

  for i, ast_str in enumerate(ast_strs):
    print(f"optimizing {i}/{len(ast_strs)}\nast={ast_str}")
    lin = ast_str_to_lin(ast_str, opts=device.renderer)
    rawbufs = bufs_from_lin(lin)
    lin = beam_search(lin, rawbufs, getenv("BEAM", 8), bool(getenv("BEAM_ESTIMATE", 1)))

    tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10)
    print(f"final time {tm*1e6:9.0f} us: {lin.colored_shape()}")
    print(lin.applied_opts)
