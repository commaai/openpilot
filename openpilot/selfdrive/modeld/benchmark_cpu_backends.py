#!/usr/bin/env python3
"""Compare steady-state CPU inference for openpilot's model in tinygrad and ONNX Runtime.

This intentionally benchmarks the policy ONNX graph only. Camera warping is a separate
tinygrad JIT in modeld and should be profiled independently when optimizing the complete
camera-to-model pipeline.
"""

import argparse
import json
import math
import os
import pathlib
import platform
import statistics
import time
from typing import Any, Callable

import numpy as np


MODELD_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = MODELD_DIR / "models" / "driving_supercombo.onnx"
DEFAULT_CACHE = MODELD_DIR.parents[2] / ".cache" / "tinygrad" / "cache.db"

INPUT_SPECS = {
  "img": ((1, 12, 128, 256), np.uint8),
  "big_img": ((1, 12, 128, 256), np.uint8),
  "features_buffer": ((1, 24, 512), np.float16),
  "desire_pulse": ((1, 25, 8), np.float16),
  "traffic_convention": ((1, 2), np.float16),
  "action_t": ((1, 2), np.float16),
}


def percentile(values: list[float], pct: float) -> float:
  ordered = sorted(values)
  index = min(len(ordered) - 1, math.ceil(pct * len(ordered)) - 1)
  return ordered[index]


def summarize(name: str, timings: list[float]) -> dict[str, Any]:
  p50 = statistics.median(timings)
  return {
    "backend": name,
    "runs": len(timings),
    "min_ms": min(timings),
    "mean_ms": statistics.fmean(timings),
    "p50_ms": p50,
    "p95_ms": percentile(timings, 0.95),
    "max_ms": max(timings),
    "fps_from_p50": 1000.0 / p50,
  }


def make_inputs(seed: int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  inputs = {}
  for name, (shape, dtype) in INPUT_SPECS.items():
    if dtype == np.uint8:
      inputs[name] = rng.integers(0, 256, size=shape, dtype=dtype)
    else:
      inputs[name] = rng.standard_normal(shape).astype(dtype)
  return inputs


def time_runner(run: Callable[[], np.ndarray], warmup: int, runs: int) -> tuple[list[float], np.ndarray]:
  output = None
  for _ in range(warmup):
    output = run()

  timings = []
  for _ in range(runs):
    start = time.perf_counter_ns()
    output = run()
    timings.append((time.perf_counter_ns() - start) / 1e6)

  assert output is not None
  return timings, output


def benchmark_onnxruntime(model: pathlib.Path, inputs: dict[str, np.ndarray], threads: int,
                          warmup: int, runs: int) -> tuple[dict[str, Any], np.ndarray]:
  try:
    import onnxruntime as ort
  except ImportError as exc:
    raise RuntimeError("ONNX Runtime is not installed; install the onnxruntime package") from exc

  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
  options.intra_op_num_threads = threads
  options.inter_op_num_threads = 1

  load_start = time.perf_counter()
  session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
  load_seconds = time.perf_counter() - load_start

  expected = {item.name: (tuple(item.shape), item.type) for item in session.get_inputs()}
  actual = {name: (value.shape, str(value.dtype)) for name, value in inputs.items()}
  if set(expected) != set(actual):
    raise RuntimeError(f"model inputs changed: expected {expected}, benchmark has {actual}")

  timings, output = time_runner(lambda: session.run(None, inputs)[0], warmup, runs)
  result = summarize(f"onnxruntime-{threads}t", timings)
  result.update({"load_seconds": load_seconds, "version": ort.__version__})
  return result, output.astype(np.float32)


def sanitize_tinygrad_environment(device: str) -> None:
  # Some shells define DEBUG or BEAM as words. tinygrad requires integer values.
  for name in ("DEBUG", "BEAM"):
    try:
      int(os.environ.get(name, "0"))
    except ValueError:
      os.environ[name] = "0"
  os.environ.setdefault("DEBUG", "0")
  os.environ.setdefault("BEAM", "0")
  os.environ["DEV"] = device
  os.environ.setdefault("JIT_BATCH_SIZE", "0")
  os.environ.setdefault("OPENPILOT_HACKS", "1")
  os.environ.setdefault("CACHEDB", str(DEFAULT_CACHE))


def benchmark_tinygrad(model: pathlib.Path, inputs: dict[str, np.ndarray], device: str,
                       warmup: int, runs: int) -> tuple[dict[str, Any], np.ndarray]:
  sanitize_tinygrad_environment(device)
  try:
    from tinygrad import Device, Tensor, TinyJit
    from tinygrad.nn.onnx import OnnxRunner
  except ImportError as exc:
    raise RuntimeError("tinygrad is not importable; add the pinned tinygrad_repo to PYTHONPATH") from exc

  load_start = time.perf_counter()
  runner = OnnxRunner(str(model))
  load_seconds = time.perf_counter() - load_start

  # Match openpilot's compile_modeld path: float16 ONNX inputs use float32 JIT
  # interfaces, while image tensors retain uint8 storage.
  tg_inputs = {
    name: Tensor(value.astype(np.float32) if value.dtype == np.float16 else value, device="NPY").realize()
    for name, value in inputs.items()
  }

  @TinyJit(prune=True)
  def run_model(**kwargs):
    values = {name: value.to(Device.DEFAULT) for name, value in kwargs.items()}
    return next(iter(runner(values).values())).cast("float32")

  def run() -> np.ndarray:
    return run_model(**tg_inputs).numpy()

  # TinyJit captures on the second call and executes the captured graph from the
  # third call onward, so ensure timing only covers the steady-state graph.
  timings, output = time_runner(run, max(warmup, 3), runs)
  result = summarize(f"tinygrad-{device.lower()}", timings)
  try:
    import tinygrad
    version = getattr(tinygrad, "__version__", "pinned-submodule")
  except Exception:
    version = "pinned-submodule"
  result.update({"load_seconds": load_seconds, "version": version})
  return result, output


def compare_outputs(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
  reference = reference.astype(np.float32).reshape(-1)
  candidate = candidate.astype(np.float32).reshape(-1)
  if reference.shape != candidate.shape:
    raise RuntimeError(f"output shape mismatch: {reference.shape} != {candidate.shape}")

  delta = np.abs(reference - candidate)
  denominator = np.maximum(np.abs(reference), 1e-6)
  cosine_denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
  cosine = float(np.dot(reference, candidate) / cosine_denominator) if cosine_denominator else 1.0
  return {
    "max_abs": float(delta.max()),
    "mean_abs": float(delta.mean()),
    "max_rel": float((delta / denominator).max()),
    "cosine_similarity": cosine,
  }


def print_result(result: dict[str, Any]) -> None:
  print(
    f"{result['backend']:24} "
    f"p50={result['p50_ms']:8.2f} ms  "
    f"p95={result['p95_ms']:8.2f} ms  "
    f"mean={result['mean_ms']:8.2f} ms  "
    f"rate={result['fps_from_p50']:6.2f} Hz"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  parser.add_argument("--backend", action="append", choices=("onnxruntime", "tinygrad"),
                      help="backend to run; may be repeated (default: both)")
  parser.add_argument("--onnxruntime-threads", type=int, nargs="+", default=[1, 2, 4])
  parser.add_argument("--tinygrad-device", default="CPU:LLVM")
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--runs", type=int, default=20)
  parser.add_argument("--seed", type=int, default=20260830)
  parser.add_argument("--json", type=pathlib.Path, help="optional path for machine-readable results")
  args = parser.parse_args()

  backends = args.backend or ["onnxruntime", "tinygrad"]
  inputs = make_inputs(args.seed)
  results: list[dict[str, Any]] = []
  outputs: dict[str, np.ndarray] = {}

  print(f"model: {args.model}")
  print(f"host: {platform.platform()} ({os.cpu_count()} logical CPUs)")
  print(f"warmup: {args.warmup}, measured runs: {args.runs}")

  if "onnxruntime" in backends:
    for threads in args.onnxruntime_threads:
      result, output = benchmark_onnxruntime(args.model, inputs, threads, args.warmup, args.runs)
      results.append(result)
      outputs[result["backend"]] = output
      print_result(result)

  if "tinygrad" in backends:
    result, output = benchmark_tinygrad(args.model, inputs, args.tinygrad_device, args.warmup, args.runs)
    results.append(result)
    outputs[result["backend"]] = output
    print_result(result)

  parity = {}
  if len(outputs) > 1:
    reference_name = next(name for name in outputs if name.startswith("onnxruntime"))
    for name, output in outputs.items():
      if name != reference_name:
        parity[f"{reference_name}_vs_{name}"] = compare_outputs(outputs[reference_name], output)
    print("parity:")
    for name, metrics in parity.items():
      print(f"  {name}: {metrics}")

  report = {
    "model": str(args.model),
    "host": platform.platform(),
    "logical_cpus": os.cpu_count(),
    "seed": args.seed,
    "results": results,
    "parity": parity,
  }
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
  main()
