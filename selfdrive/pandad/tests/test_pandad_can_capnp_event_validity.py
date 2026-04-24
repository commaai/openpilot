"""
Serialized cereal **Event.valid** flag for CAN payloads.

``docs/testing/testing-plan/TESTING-PLAN.md`` §3.1 / **R3**.

Maps: R3.
"""

from __future__ import annotations

import pytest
from cereal import log

pytest.importorskip(
  "openpilot.selfdrive.pandad.pandad_api_impl",
  reason="pandad_api_impl not built (run SCons for selfdrive/pandad)",
)


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
