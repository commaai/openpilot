"""
Smoke tests for ``selfdrive.test.support`` helpers and fixtures.

Maps: infra (LOW-LEVEL Phase 1).
"""

from __future__ import annotations

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.selfdrive.test.support.params_seed import seed_minimal_openpilot_params


def test_seed_minimal_openpilot_params_sets_toggle():
  seed_minimal_openpilot_params()
  assert Params().get("OpenpilotEnabledToggle") == b"1"


def test_openpilot_params_seeded_fixture(openpilot_params_seeded):
  assert Params().get("OpenpilotEnabledToggle") == b"1"


def test_pub_sub_factory(pub_sub_factory, monkeypatch):
  monkeypatch.setenv("SIMULATION", "1")
  pub, sub = pub_sub_factory(
    ["roadCameraState"],
    ["roadCameraState"],
    ignore_alive=["roadCameraState"],
  )
  msg = messaging.new_message("roadCameraState")
  pub.send("roadCameraState", msg)
  sub.update(2000)
  assert sub.updated["roadCameraState"]
