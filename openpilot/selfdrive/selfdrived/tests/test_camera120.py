import os
import unittest
from unittest.mock import Mock, patch

from opendbc.car.structs import car
from openpilot.common.camera120 import CAMERA120_UNUSED_SERVICES
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD, EventName, State
from openpilot.selfdrive.selfdrived.events import ET


class TestPinballCamera120(unittest.TestCase):
  def setUp(self):
    self.prefix = OpenpilotPrefix()
    self.prefix.__enter__()
    self.addCleanup(self.prefix.__exit__, None, None, None)
    env = patch.dict(os.environ, CAMERA_720P120="1")
    env.start()
    self.addCleanup(env.stop)
    params = Mock()
    params.get.return_value = None
    params.get_bool.return_value = False
    cp = car.CarParams.new_message(brand="pinball", carFingerprint="COMMA_PINBALL", notCar=True,
                                   openpilotLongitudinalControl=True, minEnableSpeed=-1, minSteerSpeed=-1)
    with patch("openpilot.selfdrive.selfdrived.selfdrived.Params", return_value=params), \
         patch("openpilot.selfdrive.selfdrived.selfdrived.get_build_metadata", return_value=Mock()):
      self.sd = SelfdriveD(cp)
    self.sd.initialized = True
    self.sd.sm.frame = 300
    for name in self.sd.sm.services:
      self.sd.sm.valid[name] = self.sd.sm.alive[name] = self.sd.sm.freq_ok[name] = name not in CAMERA120_UNUSED_SERVICES
      self.sd.sm.recv_frame[name] = 300
    self.sd.sm.data["deviceState"] = self.sd.sm["deviceState"].as_builder()
    self.sd.sm["deviceState"].freeSpacePercent = 50
    self.sd.sm["deviceState"].memoryUsagePercent = 20
    self.cs = car.CarState.new_message(canValid=True, gearShifter="drive")
    self.cs.cruiseState.enabled = True
    self.cs.cruiseState.available = True

  def test_pinball_enables_without_disabled_vision_services(self):
    self.sd.update_events(self.cs)
    self.assertNotIn(EventName.calibrationIncomplete, self.sd.events.names)
    self.assertNotIn(EventName.cameraMalfunction, self.sd.events.names)
    self.assertNotIn(EventName.commIssue, self.sd.events.names)
    self.sd.state_machine.update(self.sd.events)
    self.assertEqual(self.sd.state_machine.state, State.enabled)

  def test_missing_narrow_camera_still_blocks(self):
    self.sd.sm.alive["narrowRoadCameraState"] = False
    self.sd.update_events(self.cs)
    self.assertIn(EventName.cameraMalfunction, self.sd.events.names)
    self.assertTrue(self.sd.events.contains(ET.NO_ENTRY))

  def test_can_and_thermal_faults_still_block(self):
    self.cs.canValid = False
    self.sd.sm["deviceState"].thermalStatus = "overheated"
    self.sd.update_events(self.cs)
    self.assertIn(EventName.canError, self.sd.events.names)
    self.assertIn(EventName.overheat, self.sd.events.names)
    self.assertTrue(self.sd.events.contains(ET.NO_ENTRY))


if __name__ == "__main__":
  unittest.main()
