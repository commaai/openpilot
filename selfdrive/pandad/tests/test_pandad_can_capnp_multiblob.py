"""
Multi-blob decode path for ``can_capnp_to_list``.

``docs/testing/testing-plan/TESTING-PLAN.md`` §3.1 / **R3**.

Maps: R3.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
  "openpilot.selfdrive.pandad.pandad_api_impl",
  reason="pandad_api_impl not built (run SCons for selfdrive/pandad)",
)


def test_can_capnp_to_list_multiple_blobs():
  from openpilot.selfdrive.pandad import can_capnp_to_list, can_list_to_can_capnp

  a = [(0x10, b"\x01", 0)]
  b = [(0x20, b"\x02", 1)]
  blob_a = can_list_to_can_capnp(a, msgtype="sendcan")
  blob_b = can_list_to_can_capnp(b, msgtype="sendcan")
  decoded = can_capnp_to_list([blob_a, blob_b], msgtype="sendcan")
  assert len(decoded) == 2
  assert [(f[0], f[1], f[2]) for f in decoded[0][1]] == a
  assert [(f[0], f[1], f[2]) for f in decoded[1][1]] == b
