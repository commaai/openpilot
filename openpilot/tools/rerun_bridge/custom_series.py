"""Evaluate jotpluggler custom_python curves."""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np

from openpilot.tools.jotpluggler.math_eval import _evaluate_user_code, _resample_to_reference


def _series_dict(series: dict[str, tuple[list[float], list[float]]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
  times: dict[str, np.ndarray] = {}
  values: dict[str, np.ndarray] = {}
  for path, (t, v) in series.items():
    times[path] = np.asarray(t, dtype=np.float64)
    values[path] = np.asarray(v, dtype=np.float64)
  return times, values


def evaluate_custom_series(
  spec: dict[str, Any],
  series: dict[str, tuple[list[float], list[float]]],
  label: str,
) -> tuple[list[float], list[float]] | None:
  linked = spec.get("linked_source")
  if not linked:
    return None
  if linked not in series:
    return None

  series_t, series_v = _series_dict(series)
  ref_t = series_t[linked]
  ref_v = series_v[linked]
  if ref_t.size < 2:
    return None

  env: dict[str, Any] = {
    "np": np,
    "time": ref_t,
    "value": ref_v,
    "t": lambda path=linked: series_t[path],
    "v": lambda path=linked: series_v[path],
  }

  for path in spec.get("additional_sources", []):
    if path not in series_t:
      raise ValueError(f"Missing additional source {path} for custom series {label}")
    env[f"v{spec['additional_sources'].index(path) + 1}"] = _resample_to_reference(ref_t, series_t[path], series_v[path])

  globals_code = spec.get("globals_code", "")
  if globals_code.strip():
    exec(globals_code, env, env)

  function_code = spec.get("function_code", "")
  result = _evaluate_user_code(function_code, env)

  if isinstance(result, tuple) and len(result) == 2:
    out_t = np.asarray(result[0], dtype=np.float64).reshape(-1)
    out_v = np.asarray(result[1], dtype=np.float64).reshape(-1)
  else:
    out_v = np.asarray(result, dtype=np.float64).reshape(-1)
    out_t = ref_t[: out_v.size]

  if out_t.size < 2 or out_t.size != out_v.size:
    return None
  return out_t.tolist(), out_v.tolist()


def apply_layout_custom_series(
  layout: dict,
  series: dict[str, tuple[list[float], list[float]]],
) -> dict[str, tuple[list[float], list[float]]]:
  out = dict(series)
  for tab in layout.get("tabs", []):
    for pane in _iter_panes(tab.get("root", {})):
      for curve in pane.get("curves", []):
        spec = curve.get("custom_python")
        if not spec:
          continue
        label = curve.get("name") or curve.get("label") or "custom"
        try:
          evaluated = evaluate_custom_series(spec, out, label)
        except Exception:
          continue
        if evaluated is None:
          continue
        safe = "".join(ch if ch.isalnum() or ch in "-_./" else "_" for ch in label).strip("_")
        path = f"/custom/{safe or 'series'}"
        out[path] = evaluated
  return out


def _iter_panes(node: dict):
  if "curves" in node or node.get("kind") in {"map", "camera", "plot"}:
    yield node
  for child in node.get("children", []):
    yield from _iter_panes(child)