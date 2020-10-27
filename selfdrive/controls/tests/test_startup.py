#!/usr/bin/env python3
import time
import unittest
from parameterized import parameterized

from cereal import log, car
import cereal.messaging as messaging
from common.params import Params
from selfdrive.boardd.boardd_api_impl import can_list_to_can_capnp # pylint: disable=no-name-in-module,import-error
from selfdrive.car.fingerprints import _FINGERPRINTS
from selfdrive.car.hyundai.values import CAR as HYUNDAI
from selfdrive.car.mazda.values import CAR as MAZDA
from selfdrive.controls.lib.events import EVENT_NAME
from selfdrive.test.helpers import with_processes

EventName = car.CarEvent.EventName

class TestStartup(unittest.TestCase):

  @parameterized.expand([
    # TODO: test EventName.startup for release branches

    # officially supported car
    (EventName.startupMaster, HYUNDAI.SONATA, False),
    (EventName.startupMaster, HYUNDAI.SONATA, True),

    # community supported car
    (EventName.startupMaster, HYUNDAI.KIA_STINGER, True),
    (EventName.communityFeatureDisallowed, HYUNDAI.KIA_STINGER, False),

    # dashcamOnly car
    (EventName.startupNoControl, MAZDA.CX5, True),
    (EventName.startupNoControl, MAZDA.CX5, False),

    # unrecognized car
    (EventName.startupNoCar, None, True),
    (EventName.startupNoCar, None, False),
  ])
  @with_processes(['controlsd'])
  def test_startup_alert(self, expected_event, car, toggle_enabled):

    # TODO: this should be done without any real sockets
    controls_sock = messaging.sub_sock("controlsState")
    pm = messaging.PubMaster(['can', 'health'])

    params = Params()
    params.clear_all()
    params.put("Passive", b"0")
    params.put("OpenpilotEnabledToggle", b"1")
    params.put("CommunityFeaturesToggle", b"1" if toggle_enabled else b"0")

    time.sleep(2) # wait for controlsd to be ready

    health = messaging.new_message('health')
    health.health.hwType = log.HealthData.HwType.uno
    pm.send('health', health)

    # fingerprint
    if car is None:
      finger = {addr: 1 for addr in range(1, 100)}
    else:
      finger = _FINGERPRINTS[car][0]

    for _ in range(500):
      msgs = [[addr, 0, b'\x00'*length, 0] for addr, length in finger.items()]
      pm.send('can', can_list_to_can_capnp(msgs))

      time.sleep(0.01)
      msgs = messaging.drain_sock(controls_sock)
      if len(msgs):
        event_name = msgs[0].controlsState.alertType.split("/")[0]
        self.assertEqual(EVENT_NAME[expected_event], event_name,
                         f"expected {EVENT_NAME[expected_event]} for '{car}', got {event_name}")
        break
    else:
      self.fail(f"failed to fingerprint {car}")

if __name__ == "__main__":
  unittest.main()
