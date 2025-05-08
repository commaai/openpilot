from parameterized import parameterized

from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.nissan.values import CAR as NISSAN
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.tesla.values import CAR as TESLA
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces


FINGERPRINT_EXACT_MATCH = [HONDA.HONDA_CIVIC_BOSCH, TOYOTA.TOYOTA_RAV4_TSS2_2022, HYUNDAI.HYUNDAI_IONIQ_5]
FINGERPRINT_FUZZY_MATCH = [HONDA.HONDA_CIVIC_BOSCH_DIESEL, HYUNDAI.GENESIS_G70_2020, HYUNDAI.HYUNDAI_IONIQ_6]
FINGERPRINT_ANGLE_NO_MATCH = [TOYOTA.TOYOTA_RAV4_TSS2_2023, NISSAN.NISSAN_LEAF, TESLA.TESLA_MODEL_3]


class TestNNLCFingerprintBase:

  @staticmethod
  def _setup_platform(car_name):
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
    CI = CarInterface(CP, CP_SP)

    sunnypilot_interfaces.setup_interfaces(CI, Params())

    return CI

  @parameterized.expand(FINGERPRINT_EXACT_MATCH)
  def test_exact_fingerprint(self, car_name):
    CI = self._setup_platform(car_name)
    assert CI.CP_SP.neuralNetworkLateralControl.model.name != "MOCK" and not CI.CP_SP.neuralNetworkLateralControl.fuzzyFingerprint

  @parameterized.expand(FINGERPRINT_FUZZY_MATCH)
  def test_fuzzy_fingerprint(self, car_name):
    CI = self._setup_platform(car_name)
    assert CI.CP_SP.neuralNetworkLateralControl.model.name != "MOCK" and CI.CP_SP.neuralNetworkLateralControl.fuzzyFingerprint

  @parameterized.expand(FINGERPRINT_ANGLE_NO_MATCH)
  def test_no_fingerprint(self, car_name):
    CI = self._setup_platform(car_name)
    assert CI.CP_SP.neuralNetworkLateralControl.model.name == "MOCK"
