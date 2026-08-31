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
import tempfile
import threading
import time
from collections import defaultdict
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


class PeakMemorySampler:
  def __init__(self, enabled: bool):
    self.enabled = enabled
    self.baseline = 0
    self.peak = 0
    self.final = 0
    self._stop = threading.Event()
    self._thread = None
    self._process = None

  def __enter__(self):
    if not self.enabled:
      return self
    try:
      import psutil
    except ImportError as exc:
      raise RuntimeError("psutil is required for --measure-memory") from exc
    self._process = psutil.Process()
    self.baseline = self.peak = self._process.memory_info().rss

    def sample() -> None:
      while not self._stop.wait(0.001):
        self.peak = max(self.peak, self._process.memory_info().rss)

    self._thread = threading.Thread(target=sample, name="peak-memory-sampler", daemon=True)
    self._thread.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if not self.enabled:
      return
    self._stop.set()
    assert self._thread is not None and self._process is not None
    self._thread.join()
    self.final = self._process.memory_info().rss
    self.peak = max(self.peak, self.final)

  def report(self) -> dict[str, float]:
    mib = 1024 * 1024
    return {
      "baseline_rss_mib": self.baseline / mib,
      "peak_rss_mib": self.peak / mib,
      "peak_delta_mib": (self.peak - self.baseline) / mib,
      "final_rss_mib": self.final / mib,
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


def make_ort_options(ort, threads: int, profile_prefix: pathlib.Path | None = None):
  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
  options.intra_op_num_threads = threads
  options.inter_op_num_threads = 1
  if profile_prefix is not None:
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
  return options


def prepare_ort_inputs(session, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  type_map = {
    "tensor(uint8)": np.uint8,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
  }
  session_inputs = {}
  for item in session.get_inputs():
    if item.name not in inputs:
      raise RuntimeError(f"benchmark input {item.name!r} is missing")
    if item.type not in type_map:
      raise RuntimeError(f"unsupported ONNX Runtime input type {item.type!r} for {item.name!r}")
    value = inputs[item.name].astype(type_map[item.type], copy=False)
    if tuple(item.shape) != value.shape:
      raise RuntimeError(f"shape mismatch for {item.name}: model has {item.shape}, benchmark has {value.shape}")
    session_inputs[item.name] = value
  return session_inputs


def summarize_ort_profile(profile_path: pathlib.Path) -> dict[str, Any]:
  events = json.loads(profile_path.read_text())
  operator_totals: dict[str, dict[str, float | int]] = defaultdict(lambda: {"duration_us": 0.0, "calls": 0})
  node_totals: dict[str, dict[str, Any]] = defaultdict(lambda: {"duration_us": 0.0, "calls": 0, "operator": "unknown"})
  for event in events:
    if event.get("cat") != "Node" or "dur" not in event:
      continue
    args = event.get("args", {})
    operator = args.get("op_name", "unknown")
    node = event.get("name", "unknown")
    duration = float(event["dur"])
    operator_totals[operator]["duration_us"] += duration
    operator_totals[operator]["calls"] += 1
    node_totals[node]["duration_us"] += duration
    node_totals[node]["calls"] += 1
    node_totals[node]["operator"] = operator

  total_us = sum(float(value["duration_us"]) for value in operator_totals.values())
  def ranked(values: dict[str, dict[str, Any]], include_operator: bool) -> list[dict[str, Any]]:
    rows = []
    for name, value in values.items():
      duration_us = float(value["duration_us"])
      row = {
        "name": name,
        "calls": int(value["calls"]),
        "total_ms": duration_us / 1000.0,
        "share_pct": 100.0 * duration_us / total_us if total_us else 0.0,
      }
      if include_operator:
        row["operator"] = value["operator"]
      rows.append(row)
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)[:15]

  return {
    "total_node_ms": total_us / 1000.0,
    "top_operator_types": ranked(operator_totals, False),
    "top_nodes": ranked(node_totals, True),
  }


def profile_onnxruntime(ort, model: pathlib.Path, inputs: dict[str, np.ndarray], threads: int,
                        profile_runs: int) -> dict[str, Any]:
  with tempfile.TemporaryDirectory() as tmp:
    options = make_ort_options(ort, threads, pathlib.Path(tmp) / "ort-profile")
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    session_inputs = prepare_ort_inputs(session, inputs)
    for _ in range(profile_runs):
      session.run(None, session_inputs)
    profile_path = pathlib.Path(session.end_profiling())
    return summarize_ort_profile(profile_path)


def make_ort_runner(ort, session, session_inputs: dict[str, np.ndarray], io_binding: bool) -> Callable[[], np.ndarray]:
  if not io_binding:
    return lambda: session.run(None, session_inputs)[0]

  outputs = session.get_outputs()
  if len(outputs) != 1 or not all(isinstance(dim, int) for dim in outputs[0].shape):
    raise RuntimeError("I/O binding benchmark requires exactly one statically shaped output")
  dtype_map = {"tensor(float16)": np.float16, "tensor(float)": np.float32}
  if outputs[0].type not in dtype_map:
    raise RuntimeError(f"unsupported I/O binding output type {outputs[0].type!r}")

  output = np.empty(tuple(outputs[0].shape), dtype=dtype_map[outputs[0].type])
  binding = session.io_binding()
  for name, value in session_inputs.items():
    binding.bind_cpu_input(name, value)
  binding.bind_ortvalue_output(outputs[0].name, ort.OrtValue.ortvalue_from_numpy(output))

  def run() -> np.ndarray:
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    return output
  return run


def benchmark_onnxruntime(model: pathlib.Path, inputs: dict[str, np.ndarray], threads: int,
                          warmup: int, runs: int, profile_runs: int, measure_memory: bool,
                          io_binding: bool) -> tuple[dict[str, Any], np.ndarray]:
  try:
    import onnxruntime as ort
  except ImportError as exc:
    raise RuntimeError("ONNX Runtime is not installed; install the onnxruntime package") from exc

  memory = PeakMemorySampler(measure_memory)
  with memory:
    options = make_ort_options(ort, threads)
    load_start = time.perf_counter()
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - load_start
    session_inputs = prepare_ort_inputs(session, inputs)
    run = make_ort_runner(ort, session, session_inputs, io_binding)
    timings, output = time_runner(run, warmup, runs)

  suffix = "-iobinding" if io_binding else ""
  result = summarize(f"onnxruntime-{threads}t{suffix}", timings)
  result.update({"load_seconds": load_seconds, "version": ort.__version__})
  if measure_memory:
    result["memory"] = memory.report()
  if profile_runs:
    result["operator_profile"] = profile_onnxruntime(ort, model, inputs, threads, profile_runs)
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
                       warmup: int, runs: int, measure_memory: bool) -> tuple[dict[str, Any], np.ndarray]:
  sanitize_tinygrad_environment(device)
  try:
    from tinygrad import Device, Tensor, TinyJit
    from tinygrad.nn.onnx import OnnxRunner
  except ImportError as exc:
    raise RuntimeError("tinygrad is not importable; add the pinned tinygrad_repo to PYTHONPATH") from exc

  memory = PeakMemorySampler(measure_memory)
  with memory:
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
  if measure_memory:
    result["memory"] = memory.report()
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


def compare_onnxruntime_models(reference_model: pathlib.Path, candidate_model: pathlib.Path,
                               inputs: dict[str, np.ndarray], threads: int) -> dict[str, float]:
  try:
    import onnxruntime as ort
  except ImportError as exc:
    raise RuntimeError("ONNX Runtime is required for model comparison") from exc
  sessions = [
    ort.InferenceSession(str(model), sess_options=make_ort_options(ort, threads), providers=["CPUExecutionProvider"])
    for model in (reference_model, candidate_model)
  ]
  outputs = [session.run(None, prepare_ort_inputs(session, inputs))[0] for session in sessions]
  return compare_outputs(outputs[0], outputs[1])


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
  parser.add_argument("--compare-model", type=pathlib.Path,
                      help="optional second model for same-input ONNX Runtime output comparison")
  parser.add_argument("--backend", action="append", choices=("onnxruntime", "tinygrad"),
                      help="backend to run; may be repeated (default: both)")
  parser.add_argument("--onnxruntime-threads", type=int, nargs="+", default=[1, 2, 4])
  parser.add_argument("--tinygrad-device", default="CPU:LLVM")
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--runs", type=int, default=20)
  parser.add_argument("--profile-runs", type=int, default=0,
                      help="run a separate profiled ONNX Runtime session this many times")
  parser.add_argument("--measure-memory", action="store_true",
                      help="sample process RSS during model load, warmup, and measured runs")
  parser.add_argument("--onnxruntime-io-binding", action="store_true",
                      help="reuse pre-bound ONNX Runtime input and output buffers")
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
      result, output = benchmark_onnxruntime(
        args.model, inputs, threads, args.warmup, args.runs, args.profile_runs,
        args.measure_memory, args.onnxruntime_io_binding,
      )
      results.append(result)
      outputs[result["backend"]] = output
      print_result(result)
      if memory := result.get("memory"):
        print(f"  RSS baseline={memory['baseline_rss_mib']:.1f} MiB, "
              f"peak={memory['peak_rss_mib']:.1f} MiB, delta={memory['peak_delta_mib']:.1f} MiB")
      if profile := result.get("operator_profile"):
        print("  top operators: " + ", ".join(
          f"{item['name']}={item['share_pct']:.1f}%" for item in profile["top_operator_types"][:5]
        ))

  if "tinygrad" in backends:
    result, output = benchmark_tinygrad(
      args.model, inputs, args.tinygrad_device, args.warmup, args.runs, args.measure_memory,
    )
    results.append(result)
    outputs[result["backend"]] = output
    print_result(result)
    if memory := result.get("memory"):
      print(f"  RSS baseline={memory['baseline_rss_mib']:.1f} MiB, "
            f"peak={memory['peak_rss_mib']:.1f} MiB, delta={memory['peak_delta_mib']:.1f} MiB")

  parity = {}
  if len(outputs) > 1:
    reference_name = next(name for name in outputs if name.startswith("onnxruntime"))
    for name, output in outputs.items():
      if name != reference_name:
        parity[f"{reference_name}_vs_{name}"] = compare_outputs(outputs[reference_name], output)
    print("parity:")
    for name, metrics in parity.items():
      print(f"  {name}: {metrics}")

  model_parity = {}
  if args.compare_model:
    threads = args.onnxruntime_threads[0]
    model_parity = compare_onnxruntime_models(args.model, args.compare_model, inputs, threads)
    print(f"model parity ({args.model.name} vs {args.compare_model.name}, {threads} threads): {model_parity}")

  report = {
    "model": str(args.model),
    "host": platform.platform(),
    "logical_cpus": os.cpu_count(),
    "seed": args.seed,
    "results": results,
    "parity": parity,
    "model_parity": model_parity,
  }
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
  main()
