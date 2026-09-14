import importlib
import os
import unittest
from unittest.mock import Mock, patch

from opendbc.car.structs import car
from openpilot.common.camera120 import CAMERA120_DISABLED_PROCESSES, pinball_camera120
from openpilot.system.manager import process_config


class TestCamera120Processes(unittest.TestCase):
  def setUp(self):
    self.env = patch.dict(os.environ, CAMERA_720P120="1")
    self.env.start()
    self.config = importlib.reload(process_config)
    self.params = Mock()
    self.params.get_bool.return_value = False
    self.cp = car.CarParams.new_message(brand="pinball", notCar=True)

  def tearDown(self):
    self.env.stop()
    importlib.reload(process_config)

  def running(self, started):
    return {p.name for p in self.config.procs if p.enabled and p.should_run(started, self.params, self.cp)}

  def test_onroad_has_only_stream_encoder_and_keeps_pinball_controls(self):
    running = self.running(True)
    self.assertTrue({"camerad", "stream_encoderd", "webrtcd", "card", "pandad", "selfdrived", "joystickd", "hardwared", "ui"} <= running)
    self.assertFalse(CAMERA120_DISABLED_PROCESSES & running)

  def test_offroad_stops_cameras_encoders_and_controls(self):
    self.assertFalse({"camerad", "stream_encoderd", "webrtcd", "card", "selfdrived", "joystickd"} & self.running(False))

  def test_experiment_cannot_enable_non_pinball_controls(self):
    for brand, not_car in (("toyota", False), ("body", True), ("pinball", False)):
      self.cp.brand, self.cp.notCar = brand, not_car
      self.assertFalse(pinball_camera120(self.cp))
      self.assertFalse({"selfdrived", "joystickd", "controlsd", "joystick"} & self.running(True))

  def test_opt_out_restores_normal_process_configuration(self):
    os.environ["CAMERA_720P120"] = "0"
    self.config = importlib.reload(process_config)
    self.assertTrue({"encoderd", "loggerd", "modeld", "plannerd", "locationd"} <= self.running(True))
    self.assertFalse(pinball_camera120(self.cp))


if __name__ == "__main__":
  unittest.main()
