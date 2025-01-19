"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import messaging, custom
from opendbc.car import structs
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState


class LongitudinalPlannerSP:
  def __init__(self, CP: structs.CarParams, mpc):
    self.dec = DynamicExperimentalController(CP, mpc)

  def get_mpc_mode(self) -> str | None:
    if not self.dec.active():
      return None

    return self.dec.mode()

  def update(self, sm: messaging.SubMaster) -> None:
    self.dec.update(sm)

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP

    # Dynamic Experimental Control
    dec = longitudinalPlanSP.dec
    dec.state = DecState.blended if self.dec.mode() == 'blended' else DecState.acc
    dec.enabled = self.dec.enabled()
    dec.active = self.dec.active()

    pm.send('longitudinalPlanSP', plan_sp_send)
