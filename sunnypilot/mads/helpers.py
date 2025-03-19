"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.common.params import Params
from opendbc.car import structs
from opendbc.safety import ALTERNATIVE_EXPERIENCE
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP, HyundaiSafetyFlagsSP


def set_alternative_experience(CP: structs.CarParams, params: Params):
  enabled = params.get_bool("Mads")
  pause_lateral_on_brake = params.get_bool("MadsPauseLateralOnBrake")

  if enabled:
    CP.alternativeExperience |= ALTERNATIVE_EXPERIENCE.ENABLE_MADS

    if pause_lateral_on_brake:
      CP.alternativeExperience |= ALTERNATIVE_EXPERIENCE.DISENGAGE_LATERAL_ON_BRAKE


def set_car_specific_params(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params: Params):
  if CP.brand == "hyundai":
    # TODO-SP: This should be separated from MADS module for future implementations
    #          Use "HyundaiLongitudinalMainCruiseToggleable" param
    hyundai_cruise_main_toggleable = True
    if hyundai_cruise_main_toggleable:
      CP_SP.flags |= HyundaiFlagsSP.LONGITUDINAL_MAIN_CRUISE_TOGGLEABLE.value
      CP_SP.safetyParam |= HyundaiSafetyFlagsSP.LONG_MAIN_CRUISE_TOGGLEABLE

  # MADS is currently not supported in Tesla due to lack of consistent states to engage controls
  # TODO-SP: To enable MADS for Tesla, identify consistent signals for MADS toggling
  if CP.brand == "tesla":
    params.remove("Mads")

  # MADS is currently not supported in Rivian due to lack of consistent states to engage controls
  # TODO-SP: To enable MADS for Rivian, identify consistent signals for MADS toggling
  if CP.brand == "rivian":
    params.remove("Mads")
