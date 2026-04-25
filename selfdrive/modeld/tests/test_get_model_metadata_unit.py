"""
Unit tests for ``get_model_metadata`` helpers (no ONNX file I/O).

Maps: R1.
"""

from __future__ import annotations

from types import SimpleNamespace

from openpilot.selfdrive.modeld.get_model_metadata import get_metadata_value_by_name, get_name_and_shape


def test_get_metadata_value_by_name_hit_and_miss():
  props = [SimpleNamespace(key="output_slices", value="abc"), SimpleNamespace(key="other", value="x")]
  model = SimpleNamespace(metadata_props=props)
  assert get_metadata_value_by_name(model, "output_slices") == "abc"
  assert get_metadata_value_by_name(model, "missing") is None


def test_get_name_and_shape_reads_dims_and_name():
  d0, d1 = SimpleNamespace(dim_value=1), SimpleNamespace(dim_value=64)
  shape = SimpleNamespace(dim=[d0, d1])
  tensor_type = SimpleNamespace(shape=shape)
  t = SimpleNamespace(tensor_type=tensor_type)
  vi = SimpleNamespace(name="output0", type=t)
  name, shape_t = get_name_and_shape(vi)
  assert name == "output0"
  assert shape_t == (1, 64)
