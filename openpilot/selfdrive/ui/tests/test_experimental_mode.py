import itertools
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from opendbc.car.structs import car
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.selfdrive.car.card import Car
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD
from openpilot.selfdrive.ui.layouts.settings.developer import DeveloperLayout
from openpilot.selfdrive.ui.mici.layouts.settings.developer import DeveloperLayoutMici
from openpilot.selfdrive.ui.ui_state import ui_state


class TestExperimentalMode(unittest.TestCase):
  def setUp(self):
    self.params_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.params_dir.cleanup)
    self.params = Params(self.params_dir.name)
    # Drain pending writes before removing the temporary parameter directory.
    self.addCleanup(self.params._finalizer)

  def read_experimental_mode(self, process_class, longitudinal_control=True):
    # Exercise one iteration of the real parameter reader without starting car/control loops.
    process = process_class.__new__(process_class)
    process.params = self.params
    process.CP = car.CarParams.new_message(openpilotLongitudinalControl=longitudinal_control)
    stop = threading.Event()
    with patch("time.sleep", side_effect=lambda _: stop.set()):
      process.params_thread(stop)
    return process.experimental_mode

  def toggle_lateral_maneuvers(self, layout_class, enabled):
    # Only the visual controls are mocked; parameter writes and the callback are real.
    layout = layout_class.__new__(layout_class)
    layout._params = self.params
    layout._joystick_toggle = Mock()
    layout._long_maneuver_toggle = Mock()
    with patch.object(ui_state, "params", self.params):
      layout._on_lat_maneuver_mode(enabled)

  def test_toggle_preserves_preference(self):
    for layout, preference, enabled in itertools.product((DeveloperLayout, DeveloperLayoutMici), (False, True), (False, True)):
      with self.subTest(layout=layout.__name__, preference=preference, enabled=enabled):
        self.params.put_bool("ExperimentalMode", preference, block=True)
        self.toggle_lateral_maneuvers(layout, enabled)
        self.assertEqual(self.params.get_bool("ExperimentalMode"), preference)
        self.assertEqual(self.params.get_bool("LateralManeuverMode"), enabled)

  def test_car_initialization_suppresses_experimental_mode_during_maneuvers(self):
    self.params.put_bool("OpenpilotEnabledToggle", True, block=True)
    self.params.put_bool("ExperimentalMode", True, block=True)
    self.params.put_bool("LateralManeuverMode", True, block=True)
    interface = Mock(CP=car.CarParams.new_message(openpilotLongitudinalControl=True), CC=object())
    with patch("openpilot.selfdrive.car.card.Params", return_value=self.params), \
         patch("openpilot.selfdrive.car.card.messaging.sub_sock"), \
         patch("openpilot.selfdrive.car.card.messaging.SubMaster"), \
         patch("openpilot.selfdrive.car.card.messaging.PubMaster"):
      process = Car(CI=interface)
    self.assertFalse(process.experimental_mode)
    self.assertTrue(self.params.get_bool("ExperimentalMode"))

  def test_runtime_suppresses_experimental_mode_during_maneuvers(self):
    for process, preference, longitudinal, maneuver in itertools.product((Car, SelfdriveD), (False, True), (False, True), (False, True)):
      with self.subTest(process=process.__name__, preference=preference, longitudinal=longitudinal, maneuver=maneuver):
        self.params.put_bool("ExperimentalMode", preference, block=True)
        self.params.put_bool("LateralManeuverMode", maneuver, block=True)
        self.assertEqual(self.read_experimental_mode(process, longitudinal), preference and longitudinal and not maneuver)
        self.assertEqual(self.params.get_bool("ExperimentalMode"), preference)

  def test_preference_survives_maneuver_cleanup(self):
    for layout, preference, cleanup in itertools.product((DeveloperLayout, DeveloperLayoutMici), (False, True),
                                                        (None, ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION)):
      with self.subTest(layout=layout.__name__, preference=preference, cleanup=cleanup):
        self.params.put_bool("ExperimentalMode", preference, block=True)
        self.toggle_lateral_maneuvers(layout, True)
        for process in (Car, SelfdriveD):
          self.assertFalse(self.read_experimental_mode(process))

        if cleanup is None:
          self.toggle_lateral_maneuvers(layout, False)
        else:
          self.params.clear_all(cleanup)

        self.assertFalse(self.params.get_bool("LateralManeuverMode"))
        self.assertEqual(self.params.get_bool("ExperimentalMode"), preference)
        for process in (Car, SelfdriveD):
          self.assertEqual(self.read_experimental_mode(process), preference)


if __name__ == "__main__":
  unittest.main()
