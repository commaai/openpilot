"""
The MIT License

Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Last updated: July 29, 2024
"""

from panda import Panda, ALTERNATIVE_EXPERIENCE

from openpilot.common.params import Params

from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP


class MadsParams:
  def __init__(self):
    self.params = Params()

  def read_param(self, key: str):
    return self.params.get_bool(key)

  def set_alternative_experience(self, CP):
    enabled = self.read_param("Mads")
    pause_lateral_on_brake = self.read_param("MadsPauseLateralOnBrake")

    if enabled:
      CP.alternativeExperience |= ALTERNATIVE_EXPERIENCE.ENABLE_MADS

      if pause_lateral_on_brake:
        CP.alternativeExperience |= ALTERNATIVE_EXPERIENCE.DISENGAGE_LATERAL_ON_BRAKE

  def set_car_specific_params(self, CP):
    if CP.carName == "hyundai":
      # TODO-SP: This should be separated from MADS module for future implementations
      #          Use "HyundaiLongitudinalMainCruiseToggleable" param
      hyundai_cruise_main_toggleable = True
      if hyundai_cruise_main_toggleable:
        CP.sunnypilotFlags |= HyundaiFlagsSP.LONGITUDINAL_MAIN_CRUISE_TOGGLEABLE.value
        CP.safetyConfigs[-1].safetyParam |= Panda.FLAG_HYUNDAI_LONG_MAIN_CRUISE_TOGGLEABLE

    # MADS is currently not supported in Tesla due to lack of consistent states to engage controls
    # TODO-SP: To enable MADS for Tesla, identify consistent signals for MADS toggling
    if CP.carName == "tesla":
      self.params.remove("Mads")
