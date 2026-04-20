"""
Smoke tests for ``system.tests.support`` helpers and fixtures.

Maps: infra (LOW-LEVEL Phase 1); R4 when messaging round-trip is exercised.
"""

from __future__ import annotations

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.system.tests.support.messaging import make_pub_sub
from openpilot.system.tests.support.params_seed import seed_full_stack_params, seed_system_daemon_params


def test_seed_system_daemon_params_sets_expected_keys():
  seed_system_daemon_params()
  p = Params()
  assert p.get("IsOffroad") == b"1"
  assert p.get("DongleId") == b"0000000000000000"


def test_seed_full_stack_params_sets_baseline():
  seed_full_stack_params()
  p = Params()
  assert p.get("IsOffroad") == b"1"
  assert p.get("OpenpilotEnabledToggle") == b"1"


def test_system_daemon_params_fixture(system_daemon_params):
  assert Params().get("IsOffroad") == b"1"


def test_system_full_stack_params_fixture(system_full_stack_params):
  assert Params().get("OpenpilotEnabledToggle") == b"1"


def test_make_pub_sub_helper(monkeypatch):
  monkeypatch.setenv("SIMULATION", "1")
  pub, sub = make_pub_sub(["roadCameraState"], ["roadCameraState"], ignore_alive=["roadCameraState"])
  msg = messaging.new_message("roadCameraState")
  pub.send("roadCameraState", msg)
  sub.update(2000)
  assert sub.updated["roadCameraState"]


def test_system_pub_sub_factory(system_pub_sub_factory, monkeypatch):
  monkeypatch.setenv("SIMULATION", "1")
  pub, sub = system_pub_sub_factory(
    ["roadCameraState"],
    ["roadCameraState"],
    ignore_alive=["roadCameraState"],
  )
  msg = messaging.new_message("roadCameraState")
  pub.send("roadCameraState", msg)
  sub.update(2000)
  assert sub.updated["roadCameraState"]
