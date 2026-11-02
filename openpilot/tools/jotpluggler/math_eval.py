#!/usr/bin/env python3

import json
import sys
import textwrap
import traceback

import numpy as np


def _load_manifest(path: str) -> dict:
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def _load_vector(path: str) -> np.ndarray:
  return np.fromfile(path, dtype=np.float64)


def _write_vector(path: str, values: np.ndarray) -> None:
  np.asarray(values, dtype=np.float64).tofile(path)


def _resample_to_reference(ref_t: np.ndarray, src_t: np.ndarray, src_v: np.ndarray) -> np.ndarray:
  ref_t = np.asarray(ref_t, dtype=np.float64).reshape(-1)
  src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
  src_v = np.asarray(src_v, dtype=np.float64).reshape(-1)
  if ref_t.size == 0 or src_t.size == 0 or src_v.size == 0:
    return np.empty_like(ref_t)
  indices = np.searchsorted(src_t, ref_t, side="right") - 1
  indices = np.clip(indices, 0, src_v.size - 1)
  return src_v[indices]


def _evaluate_user_code(code: str, env: dict):
  stripped = code.strip()
  if not stripped:
    raise ValueError("Function body is empty")

  expr = stripped
  if expr.startswith("return "):
    expr = expr[7:].strip()
  try:
    return eval(expr, env, env)
  except SyntaxError:
    pass

  function_src = "def __jotpluggler_eval__():\n" + textwrap.indent(code, "    ")
  exec(function_src, env, env)
  return env["__jotpluggler_eval__"]()


def main() -> int:
  if len(sys.argv) != 6:
    print("usage: math_eval.py <manifest.json> <globals.py> <code.py> <out_t.bin> <out_v.bin>", file=sys.stderr)
    return 2

  manifest_path, globals_path, code_path, out_t_path, out_v_path = sys.argv[1:6]
  manifest = _load_manifest(manifest_path)

  series_t = {}
  series_v = {}
  for entry in manifest.get("series", []):
    path = entry["path"]
    series_t[path] = _load_vector(entry["t"])
    series_v[path] = _load_vector(entry["v"])

  first_path = manifest.get("linked_source") or None

  def remember(path: str) -> None:
    nonlocal first_path
    if first_path is None:
      first_path = path

  def t(path: str) -> np.ndarray:
    remember(path)
    return series_t[path]

  def v(path: str) -> np.ndarray:
    remember(path)
    return series_v[path]

  additional_sources = list(manifest.get("additional_sources", []))
  linked_source = manifest.get("linked_source") or ""
  paths = list(manifest.get("paths", []))

  env = {
    "__builtins__": __builtins__,
    "np": np,
    "t": t,
    "v": v,
    "paths": paths,
    "linked_source": linked_source,
    "additional_sources": additional_sources,
  }

  reference_time = None
  if linked_source:
    reference_time = series_t[linked_source]
    env["time"] = reference_time
    env["value"] = series_v[linked_source]

  for i, path in enumerate(additional_sources, start=1):
    if reference_time is None:
      env[f"t{i}"] = series_t[path]
      env[f"v{i}"] = series_v[path]
    else:
      env[f"t{i}"] = reference_time
      env[f"v{i}"] = _resample_to_reference(reference_time, series_t[path], series_v[path])

  with open(globals_path, encoding="utf-8") as f:
    globals_code = f.read()
  if globals_code.strip():
    exec(globals_code, env, env)

  with open(code_path, encoding="utf-8") as f:
    user_code = f.read()
  result = _evaluate_user_code(user_code, env)

  if isinstance(result, tuple) and len(result) == 2:
    result_t, result_v = result
  else:
    if first_path is None:
      raise ValueError("No reference series found. Set an input timeseries or return (times, values).")
    result_t = series_t[first_path]
    result_v = result

  result_t = np.asarray(result_t, dtype=np.float64).reshape(-1)
  result_v = np.asarray(result_v, dtype=np.float64).reshape(-1)
  if result_t.size == 0 or result_v.size == 0:
    raise ValueError("Custom series returned an empty result")
  if result_t.shape != result_v.shape:
    raise ValueError(f"Time/value arrays must have the same shape, got {result_t.shape} and {result_v.shape}")

  _write_vector(out_t_path, result_t)
  _write_vector(out_v_path, result_v)
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except Exception as err:
    traceback.print_exc()
    raise SystemExit(1) from err
