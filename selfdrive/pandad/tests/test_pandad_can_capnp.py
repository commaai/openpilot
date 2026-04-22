"""
Unit tests for CAN list serialization (``pandad_api_impl`` / ``can_list_to_can_capnp.cc``).

Stakeholder plan: ``docs/testing/testing-plan/TESTING-PLAN.md`` §3.1 (unit tests
for transport serialization), §3.3 (white-box boundary), and risk **R3**
(``pandad_api_impl.pyx`` + native packing).

Requires a built ``pandad_api_impl`` extension; skipped when the tree is not
compiled.

Maps: R3.
"""

from __future__ import annotations

import pytest
from cereal import log

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


def test_sendcan_valid_flag_false_on_event():
  from openpilot.selfdrive.pandad import can_list_to_can_capnp

  blob = can_list_to_can_capnp([(1, b"x", 0)], msgtype="sendcan", valid=False)
  with log.Event.from_bytes(blob) as msg:
    assert msg.valid is False


def test_sendcan_valid_flag_true_on_event():
  from openpilot.selfdrive.pandad import can_list_to_can_capnp

  blob = can_list_to_can_capnp([(1, b"x", 0)], msgtype="sendcan", valid=True)
  with log.Event.from_bytes(blob) as msg:
    assert msg.valid is True


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
