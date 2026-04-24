"""
CAN capnp **round-trip** serialization (``pandad_api_impl`` / ``can_list_to_can_capnp.cc``).

Split from the former monolithic capnp test module for clarity.

``docs/testing/testing-plan/TESTING-PLAN.md`` §3.1 unit / **R3**.

Maps: R3.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
  "openpilot.selfdrive.pandad.pandad_api_impl",
  reason="pandad_api_impl not built (run SCons for selfdrive/pandad)",
)


@pytest.mark.parametrize(
  "frames",
  [
    [(0, b"", 0)],
    [(0x7FF, b"\xff" * 8, 2)],
    [(0x100 + i, bytes([i & 0xFF]), i % 3) for i in range(16)],
    [(0x201, b"\xaa\xbb", 0), (0x202, b"\x00\x01\x02\x03\x04\x05\x06\x07", 1)],
  ],
)
def test_sendcan_roundtrip_parameterized(frames):
  from openpilot.selfdrive.pandad import can_capnp_to_list, can_list_to_can_capnp

  blob = can_list_to_can_capnp(frames, msgtype="sendcan", valid=True)
  decoded = can_capnp_to_list([blob], msgtype="sendcan")
  assert len(decoded) == 1
  _nanos, out_frames = decoded[0]
  assert len(out_frames) == len(frames)
  for got, exp in zip(out_frames, frames, strict=True):
    assert got[0] == exp[0]
    assert got[1] == exp[1]
    assert got[2] == exp[2]


def test_sendcan_empty_list_roundtrip():
  from openpilot.selfdrive.pandad import can_capnp_to_list, can_list_to_can_capnp

  blob = can_list_to_can_capnp([], msgtype="sendcan", valid=True)
  decoded = can_capnp_to_list([blob], msgtype="sendcan")
  assert len(decoded) == 1
  _nanos, out_frames = decoded[0]
  assert out_frames == []


def test_can_list_to_capnp_and_back_sendcan():
  from openpilot.selfdrive.pandad import can_capnp_to_list, can_list_to_can_capnp

  frames = [(0x201, b"\xaa\xbb", 0), (0x7FF, b"\x00", 2)]
  blob = can_list_to_can_capnp(frames, msgtype="sendcan", valid=True)
  assert isinstance(blob, bytes)
  assert len(blob) > 0

  decoded = can_capnp_to_list([blob], msgtype="sendcan")
  assert len(decoded) == 1
  _nanos, out_frames = decoded[0]
  assert len(out_frames) == 2
  assert out_frames[0][0] == 0x201
  assert out_frames[0][1] == b"\xaa\xbb"
  assert out_frames[0][2] == 0
  assert out_frames[1][0] == 0x7FF
  assert out_frames[1][1] == b"\x00"
  assert out_frames[1][2] == 2


def test_can_list_to_capnp_and_back_can():
  from openpilot.selfdrive.pandad import can_capnp_to_list, can_list_to_can_capnp

  frames = [(0x123, b"\x01\x02\x03\x04\x05\x06\x07\x08", 1)]
  blob = can_list_to_can_capnp(frames, msgtype="can", valid=True)
  decoded = can_capnp_to_list([blob], msgtype="can")
  assert len(decoded) == 1
  _nanos, out_frames = decoded[0]
  assert len(out_frames) == 1
  assert out_frames[0][0] == 0x123
  assert out_frames[0][1] == frames[0][1]
  assert out_frames[0][2] == 1
