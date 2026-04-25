"""
Multi-service pub/sub using ``make_pub_sub`` / ``system_pub_sub_factory``.

Extends harness coverage without touching subsystem tests under ``system/manager`` etc.

``docs/testing/testing-plan/TESTING-PLAN.md`` infrastructure for **R4**-style IPC checks.

Maps: R4 (messaging contract).
"""

from __future__ import annotations

import cereal.messaging as messaging


def test_pub_sub_two_services_roundtrip(system_pub_sub_factory, monkeypatch):
  monkeypatch.setenv("SIMULATION", "1")
  pubs = ["roadCameraState", "wideRoadCameraState"]
  subs = ["roadCameraState", "wideRoadCameraState"]
  ignore = subs
  pub, sub = system_pub_sub_factory(pubs, subs, ignore_alive=ignore)

  m1 = messaging.new_message("roadCameraState")
  m2 = messaging.new_message("wideRoadCameraState")
  pub.send("roadCameraState", m1)
  pub.send("wideRoadCameraState", m2)
  sub.update(3000)

  assert sub.updated["roadCameraState"]
  assert sub.updated["wideRoadCameraState"]
