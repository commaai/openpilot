#!/usr/bin/env python3
import time
import unittest
from parameterized import parameterized

from cereal import log, car
import cereal.messaging as messaging
from common.params import Params
from selfdrive.boardd.boardd_api_impl import can_list_to_can_capnp # pylint: disable=no-name-in-module,import-error
from selfdrive.car.fingerprints import _FINGERPRINTS
from selfdrive.car.toyota.values import CAR as TOYOTA
from selfdrive.car.mazda.values import CAR as MAZDA
from selfdrive.controls.lib.events import EVENT_NAME
from selfdrive.test.helpers import with_processes

EventName = car.CarEvent.EventName
Ecu = car.CarParams.Ecu

COROLLA_FW_VERSIONS = [
  (Ecu.engine, 0x7e0, None, b'\x0230ZC2000\x00\x00\x00\x00\x00\x00\x00\x0050212000\x00\x00\x00\x00\x00\x00\x00\x00'),
  (Ecu.abs, 0x7b0, None, b'F152602190\x00\x00\x00\x00\x00\x00'),
  (Ecu.eps, 0x7a1, None, b'8965B02181\x00\x00\x00\x00\x00\x00'),
  (Ecu.fwdRadar, 0x750, 0xf, b'8821F4702100\x00\x00\x00\x00'),
  (Ecu.fwdCamera, 0x750, 0x6d, b'8646F0201101\x00\x00\x00\x00'),
  (Ecu.dsu, 0x791, None, b'881510201100\x00\x00\x00\x00'),
]
COROLLA_FW_VERSIONS_FUZZY = COROLLA_FW_VERSIONS[:-1] + [(Ecu.dsu, 0x791, None, b'xxxxxx')]
COROLLA_FW_VERSIONS_NO_DSU = COROLLA_FW_VERSIONS[:-1]

CX5_FW_VERSIONS = [
  (Ecu.engine, 0x7e0, None, b'PYNF-188K2-F\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
  (Ecu.abs, 0x760, None, b'K123-437K2-E\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
  (Ecu.eps, 0x730, None, b'KJ01-3210X-G-00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
  (Ecu.fwdRadar, 0x764, None, b'K123-67XK2-F\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
  (Ecu.fwdCamera, 0x706, None, b'B61L-67XK2-T\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
  (Ecu.transmission, 0x7e1, None, b'PYNC-21PS1-B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
]

class TestStartup(unittest.TestCase):

  @parameterized.expand([
    # TODO: test EventName.startup for release branches

    # officially supported car
    (EventName.startupMaster, TOYOTA.COROLLA, COROLLA_FW_VERSIONS, "toyota"),
    (EventName.startupMaster, TOYOTA.COROLLA, COROLLA_FW_VERSIONS, "toyota"),

    # dashcamOnly car
    (EventName.startupNoControl, MAZDA.CX5, CX5_FW_VERSIONS, "mazda"),
    (EventName.startupNoControl, MAZDA.CX5, CX5_FW_VERSIONS, "mazda"),

    # unrecognized car with no fw
    (EventName.startupNoFw, None, None, ""),
    (EventName.startupNoFw, None, None, ""),

    # unrecognized car
    (EventName.startupNoCar, None, COROLLA_FW_VERSIONS[:1], "toyota"),
    (EventName.startupNoCar, None, COROLLA_FW_VERSIONS[:1], "toyota"),

    # fuzzy match
    (EventName.startupMaster, TOYOTA.COROLLA, COROLLA_FW_VERSIONS_FUZZY, "toyota"),
    (EventName.startupMaster, TOYOTA.COROLLA, COROLLA_FW_VERSIONS_FUZZY, "toyota"),
  ])
  @with_processes(['controlsd'])
  def test_startup_alert(self, expected_event, car_model, fw_versions, brand):

    # TODO: this should be done without any real sockets
    controls_sock = messaging.sub_sock("controlsState")
    pm = messaging.PubMaster(['can', 'pandaStates'])

    params = Params()
    params.clear_all()
    params.put_bool("Passive", False)
    params.put_bool("OpenpilotEnabledToggle", True)

    # Build capnn version of FW array
    if fw_versions is not None:
      car_fw = []
      cp = car.CarParams.new_message()
      for ecu, addr, subaddress, version in fw_versions:
        f = car.CarParams.CarFw.new_message()
        f.ecu = ecu
        f.address = addr
        f.fwVersion = version
        f.brand = brand

        if subaddress is not None:
          f.subAddress = subaddress

        car_fw.append(f)
      cp.carVin = "1" * 17
      cp.carFw = car_fw
      params.put("CarParamsCache", cp.to_bytes())

    time.sleep(2) # wait for controlsd to be ready

    pm.send('can', can_list_to_can_capnp([[0, 0, b"", 0]]))
    time.sleep(0.1)

    msg = messaging.new_message('pandaStates', 1)
    msg.pandaStates[0].pandaType = log.PandaState.PandaType.uno
    pm.send('pandaStates', msg)

    # fingerprint
    if (car_model is None) or (fw_versions is not None):
      finger = {addr: 1 for addr in range(1, 100)}
    else:
      finger = _FINGERPRINTS[car_model][0]

    for _ in range(1000):
      msgs = [[addr, 0, b'\x00'*length, 0] for addr, length in finger.items()]
      pm.send('can', can_list_to_can_capnp(msgs))

      time.sleep(0.01)
      msgs = messaging.drain_sock(controls_sock)
      if len(msgs):
        event_name = msgs[0].controlsState.alertType.split("/")[0]
        self.assertEqual(EVENT_NAME[expected_event], event_name,
                         f"expected {EVENT_NAME[expected_event]} for '{car_model}', got {event_name}")
        break
    else:
      self.fail(f"failed to fingerprint {car_model}")

if __name__ == "__main__":
  unittest.main()
