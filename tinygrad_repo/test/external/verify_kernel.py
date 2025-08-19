import argparse
from collections import defaultdict
from extra.optimization.helpers import kern_str_to_lin, time_linearizer
from test.external.fuzz_linearizer import compare_linearizer
from tinygrad.helpers import colored
from tinygrad.codegen.opt.kernel import Kernel

# Use this with the LOGKERNS options to verify that all executed kernels are valid and evaluate to the same ground truth results

# Example for GPT2:
# 1) Run the model to log all kernels: `PYTHONPATH=. LOGKERNS=/tmp/gpt2_kerns.txt JIT=1 HALF=1 BEAM=2 CACHELEVEL=0 python3 examples/gpt2.py --count 10 --temperature 0 --timing`   # noqa: E501
# 2) Validate the kernel correctness: `PYTHONPATH=. python3 ./test/external/verify_kernel.py --file /tmp/gpt2_kerns.txt`

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verify the correctness of one or more kernel", formatter_class=argparse.ArgumentDefaultsHelpFormatter)    # noqa: E501
  parser.add_argument("--kernel", type=str, default=None, help="a string of a tuple of (ast, applied_opts,)")
  parser.add_argument("--file", type=str, default=None, help="a file containing a tuple of ast and applied_opts, one per line")
  parser.add_argument("--pkl", type=str, default=None, help="a pickle file containing a single tuple of ast and applied_opts")
  parser.add_argument("--rtol", type=float, default=1e-2, help="relative tolerance for numerical comparison")
  parser.add_argument("--atol", type=float, default=1e-2, help="absolute tolerance for numerical comparison")
  parser.add_argument("--timing", action='store_true', help="show final timing for the kernel")
  parser.add_argument("--expected-failures", type=int, default=0, help="the number of expected failed kernels")
  args = parser.parse_args()

  if args.kernel is not None:
    print("loading kernel from args")
    test_lins = [kern_str_to_lin(args.kernel)]
  elif args.file is not None:
    print(f"loading kernel from file '{args.file}'")
    with open(args.file, 'r') as file:
      kern_strs = file.readlines()
      test_lins = [kern_str_to_lin(kern_str) for kern_str in kern_strs]
  elif args.pkl is not None:
    print(f"loading kernel from pickle file '{args.file}'")
    import pickle
    with open(args.pkl, 'rb') as file:
      (ast, applied_opts,) = pickle.load(file)
      lin = Kernel(ast)
      lin.apply_opts(applied_opts)
      test_lins = [lin]

  else:
    raise RuntimeError("no kernel specified; use --kernel, --file, or --pkl options")

  print(f"verifying {len(test_lins)} kernels")

  failed_ids = []
  failures = defaultdict(list)
  for i, test_lin in enumerate(test_lins):
    print(f"testing kernel {i}")
    print(test_lin.ast)
    print(test_lin.applied_opts)
    unoptimized_lin = Kernel(test_lin.ast)
    print(f"{unoptimized_lin.colored_shape()} -> {test_lin.colored_shape()}")
    (msg,rb,vv,gt) = compare_linearizer(test_lin, None, None, None, rtol=args.rtol, atol=args.atol)
    if msg != "PASS":
      failed_ids.append(i)
      failures[msg].append((test_lin.ast, test_lin.applied_opts))
    if args.timing:
      tm = time_linearizer(test_lin, rb, allow_test_size=False, cnt=10)
      print(f"final time {tm*1e6:9.0f} us")

  for msg, errors in failures.items():
    for i, (ast, opts) in enumerate(errors):
      print(f"{msg} {i} AST: {ast}")
      print(f"{msg} {i} OPTS: {opts}\n")

  print(f"tested {len(test_lins)} kernels")
  if failures:
    print(f"{failed_ids=}")
    for msg, errors in failures.items():
      print(f"{msg}: {len(errors)}")
    if len(failed_ids) == args.expected_failures:
      print(colored(f"{len(failed_ids)} failed as expected", "yellow"))
  if len(failed_ids) != args.expected_failures:
    raise RuntimeError(f"failed on {len(failed_ids)} kernels, expected {args.expected_failures}")
  else:
    print(colored("all passed", "green"))
