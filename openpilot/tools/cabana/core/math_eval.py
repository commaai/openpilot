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


class LuaEvaluator:
  """Use the installed Lua C runtime without rewriting the user's program."""

  def __init__(self, globals_code, function_code, additional_count):
    import ctypes
    import ctypes.util

    self.ctypes = ctypes
    library = next((ctypes.util.find_library(name) for name in ("lua5.4", "lua5.3", "lua") if ctypes.util.find_library(name)), None)
    if library is None:
      raise RuntimeError("Lua runtime unavailable. Install liblua5.4-0 (Ubuntu) or lua (Homebrew) for legacy XML equations.")
    self.lua = ctypes.CDLL(library)
    api = {
      "luaL_newstate": (ctypes.c_void_p, []),
      "luaL_openlibs": (None, [ctypes.c_void_p]),
      "lua_close": (None, [ctypes.c_void_p]),
      "luaL_loadstring": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
      "lua_pcallk": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ssize_t, ctypes.c_void_p]),
      "lua_getglobal": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
      "lua_pushnumber": (None, [ctypes.c_void_p, ctypes.c_double]),
      "lua_type": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
      "lua_toboolean": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
      "lua_tonumberx": (ctypes.c_double, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]),
      "lua_tolstring": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]),
      "lua_settop": (None, [ctypes.c_void_p, ctypes.c_int]),
    }
    for name, (result, arguments) in api.items():
      function = getattr(self.lua, name)
      function.restype, function.argtypes = result, arguments
    self.state = self.lua.luaL_newstate()
    if not self.state:
      raise MemoryError("Cannot allocate Lua state")
    try:
      self.lua.luaL_openlibs(self.state)
      self.execute(globals_code)
      parameters = ", ".join(["time", "value"] + [f"v{i + 1}" for i in range(additional_count)])
      self.execute(f"function __cabana_eval({parameters})\n{function_code}\nend")
    except BaseException:
      self.close()
      raise

  def error(self):
    size = self.ctypes.c_size_t()
    pointer = self.lua.lua_tolstring(self.state, -1, self.ctypes.byref(size))
    message = self.ctypes.string_at(pointer, size.value).decode("utf-8", errors="replace") if pointer else "Unknown Lua error"
    self.lua.lua_settop(self.state, 0)
    raise RuntimeError(message)

  def execute(self, program):
    if self.lua.luaL_loadstring(self.state, program.encode()) or self.lua.lua_pcallk(self.state, 0, 0, 0, 0, None):
      self.error()

  def sample(self, *values):
    self.lua.lua_getglobal(self.state, b"__cabana_eval")
    for value in values:
      self.lua.lua_pushnumber(self.state, float(value))
    if self.lua.lua_pcallk(self.state, len(values), 1, 0, 0, None):
      self.error()
    kind = self.lua.lua_type(self.state, -1)
    if kind == 0:
      result = float("nan")
    elif kind == 1:
      result = float(self.lua.lua_toboolean(self.state, -1))
    else:
      valid = self.ctypes.c_int()
      result = self.lua.lua_tonumberx(self.state, -1, self.ctypes.byref(valid))
      if not valid.value:
        raise ValueError("Lua equation must return a number, boolean or nil")
    self.lua.lua_settop(self.state, 0)
    return result

  def close(self):
    if self.state:
      self.lua.lua_close(self.state)
      self.state = None


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
  language = manifest.get("language", "python")
  if language == "python" and globals_code.strip():
    exec(globals_code, env, env)

  with open(code_path, encoding="utf-8") as f:
    user_code = f.read()
  if language == "lua":
    evaluator = LuaEvaluator(globals_code, user_code, len(additional_sources))
    try:
      result = np.array(
        [
          evaluator.sample(reference_time[i], env["value"][i], *(env[f"v{j + 1}"][i] for j in range(len(additional_sources))))
          for i in range(len(reference_time))
        ],
        dtype=np.float64,
      )
    finally:
      evaluator.close()
  elif language == "python":
    result = _evaluate_user_code(user_code, env)
  else:
    raise ValueError("Unknown calculation language: " + language)

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
