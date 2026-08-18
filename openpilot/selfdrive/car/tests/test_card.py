import unittest
from unittest import mock

from opendbc.car.structs import car

from openpilot.selfdrive.car.card import reset_calibration_if_car_changed


class TestCarParamsPersistence(unittest.TestCase):
  def test_calibration_reset_when_car_changes(self):
    params = mock.Mock()
    previous_CP = car.CarParams(carFingerprint="HONDA CIVIC 2016")
    current_CP = car.CarParams(carFingerprint="TOYOTA COROLLA TSS2 2019")

    reset_calibration_if_car_changed(params, previous_CP.to_bytes(), current_CP)

    params.remove.assert_called_once_with("CalibrationParams")

  def test_calibration_preserved_for_same_car(self):
    params = mock.Mock()
    previous_CP = car.CarParams(carFingerprint="HONDA CIVIC 2016")
    current_CP = car.CarParams(carFingerprint=previous_CP.carFingerprint)

    reset_calibration_if_car_changed(params, previous_CP.to_bytes(), current_CP)

    params.remove.assert_not_called()
