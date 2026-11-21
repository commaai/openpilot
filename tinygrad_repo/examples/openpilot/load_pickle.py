import argparse, time
from contextlib import nullcontext
import numpy as np
from tinygrad import Device
from tinygrad.engine.realize import lower_and_compile
from extra.bench_log import WallTimeEvent, BenchEvent
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_parameters
from examples.openpilot.helpers import allocate_inputs, load_pickle


def make_inputs(artifact, seed=100):
  rng = np.random.default_rng(seed)
  def initialize(views):
    for value in views.values():
      value[...] = (rng.standard_normal(value.shape) * 8 if np.issubdtype(value.dtype, np.floating) else
                    rng.integers(0, 2 if value.dtype == np.bool_ else 16, value.shape))
  inputs = allocate_inputs(artifact['input_specs'], initialize)
  if 'output_specs' in artifact: inputs['output_buffers'] = allocate_inputs(artifact['output_specs'])
  return inputs


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Benchmark loading or running compiled model and warp pickles.")
  parser.add_argument('pickle', nargs='?', default='/tmp/openpilot.pkl')
  parser.add_argument('--run', action='store_true', help='benchmark inference instead of loading')
  parser.add_argument('--out-of-band', action='store_true', default=bool(getenv('PICKLE_OOB')))
  parser.add_argument('--runs', type=int, help='defaults to 10 loads or 20 inference runs')
  parser.add_argument('--retarget', action='store_true', help='retarget loaded jit (requires jit to have been compiled with --retargetable)')
  args = parser.parse_args()
  if not args.run:
    load_times = []
    for _ in range(args.runs or 10):
      with WallTimeEvent(BenchEvent.STEP) as wte: load_pickle(args.pickle, out_of_band=args.out_of_band)
      load_times.append(wte.time)
      print(f"pickle load: {wte.time:6.2f} s")
    if (limit := getenv("ASSERT_MIN_LOAD_TIME", 0.0)):
      assert min(load_times) < limit, f"Speed regression, expected < {limit} s but took {min(load_times)} s"
  else:
    artifact = load_pickle(args.pickle, out_of_band=args.out_of_band)
    if args.retarget: artifact['run'].captured._linear = lower_and_compile(artifact['run'].captured._linear)
    inputs = make_inputs(artifact)
    times = []
    for _ in range(args.runs or 20):
      start = time.perf_counter()
      with WallTimeEvent(BenchEvent.STEP) if getenv('BENCHMARK_LOG', '') else nullcontext():
        output = artifact['run'](**inputs)
        enqueued = time.perf_counter()
        for device in {tensor.device for tensor in get_parameters((inputs, output))}: Device[device].synchronize()
      times.append((time.perf_counter() - start) * 1e3)
      print(f"enqueue {(enqueued-start)*1e3:6.2f} ms -- total run {times[-1]:6.2f} ms")
    if (limit := getenv("ASSERT_MIN_STEP_TIME", 0.0)):
      assert min(times) < limit, f"Speed regression, expected < {limit} ms but took {min(times)} ms"
