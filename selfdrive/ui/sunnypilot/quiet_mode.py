"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from cereal import car

from openpilot.common.params import Params

AudibleAlert = car.CarControl.HUDControl.AudibleAlert

ALERTS_ALWAYS_PLAY = {
  AudibleAlert.warningSoft,
  AudibleAlert.warningImmediate,
  AudibleAlert.promptDistracted,
  AudibleAlert.promptRepeat,
}

class QuietMode:
  def __init__(self):
    self.params = Params()
    self.enabled: bool = self.params.get_bool("QuietMode")
    self._frame = 0

  def load_param(self) -> None:
    self._frame += 1
    if self._frame % 50 == 0:  # 2.5 seconds
      self.enabled = self.params.get_bool("QuietMode")

  def should_play_sound(self, current_alert: int) -> bool:
    """
    Check if a sound should be played based on the Quiet Mode setting
    and the current alert.
    """
    if not self.enabled:
      return bool(current_alert != AudibleAlert.none)

    return current_alert in ALERTS_ALWAYS_PLAY
