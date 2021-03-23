from common.numpy_fast import clip, interp
from selfdrive.car.tesla.teslacan import TeslaCAN
from opendbc.can.packer import CANPacker
from selfdrive.car.tesla.values import CANBUS, CarControllerParams, CAR
from selfdrive.controls.lib.longitudinal_planner import calc_cruise_accel_limits,limit_accel_in_turns
import cereal.messaging as messaging

def _is_present(lead):
  return bool((not (lead is None)) and (lead.dRel > 0))

class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.last_angle = 0
    self.packer = CANPacker(dbc_name)
    self.tesla_can = TeslaCAN(dbc_name, self.packer)
    self.prev_das_steeringControl_counter = -1
    if CP.openpilotLongitudinalControl:
      self.lP = messaging.sub_sock('longitudinalPlan')
      self.rS = messaging.sub_sock('radarState')
      self.v_target = None
      self.lead_1 = None
      self.long_control_counter = 0


  def update(self, enabled, CS, frame, actuators, cruise_cancel):
    can_sends = []

    # Temp disable steering on a hands_on_fault, and allow for user override
    # TODO: better user blending
    hands_on_fault = (CS.steer_warning == "EAC_ERROR_HANDS_ON" and CS.hands_on_level >= 3)
    lkas_enabled = enabled and (not hands_on_fault)

    if lkas_enabled:
      apply_angle = actuators.steeringAngleDeg

      # Angular rate limit based on speed
      steer_up = (self.last_angle * apply_angle > 0. and abs(apply_angle) > abs(self.last_angle))
      rate_limit = CarControllerParams.RATE_LIMIT_UP if steer_up else CarControllerParams.RATE_LIMIT_DOWN
      max_angle_diff = interp(CS.out.vEgo, rate_limit.speed_points, rate_limit.max_angle_diff_points)
      apply_angle = clip(apply_angle, (self.last_angle - max_angle_diff), (self.last_angle + max_angle_diff))

      # To not fault the EPS
      apply_angle = clip(apply_angle, (CS.out.steeringAngleDeg - 20), (CS.out.steeringAngleDeg + 20))
    else:
      apply_angle = CS.out.steeringAngleDeg

    self.last_angle = apply_angle

    #only send the DAS_steeringControl after we received a new counter for it
    if enabled:
      can_sends.append(self.tesla_can.create_steering_control(apply_angle, lkas_enabled, frame))

    if enabled and self.CP.openpilotLongitudinalControl and (frame %2 == 0):
      #we use the same logic from planner here to get the speed
      long_plan = messaging.recv_one_or_none(self.lP)
      radar_state = messaging.recv_one_or_none(self.rS)
      if radar_state is not None:
        self.lead_1 = radar_state.radarState.leadOne
      if long_plan is not None:
        self.v_target = long_plan.longitudinalPlan.vTargetFuture # to try vs vTarget
        self.a_target = abs(long_plan.longitudinalPlan.aTarget) # to try vs aTarget
      if self.v_target is None:
        self.v_target = CS.out.vEgo
        self.a_target = 1
      following = False
      #TODO: see what works best for these
      test_accel_limits = [-2*self.a_target,2*self.a_target]
      test_jerk_limits = [-4*self.a_target,4*self.a_target]
      if _is_present(self.lead_1):
        following = self.lead_1.status and self.lead_1.dRel < 45.0 and self.lead_1.vLeadK > CS.out.vEgo and self.lead_1.aLeadK > 0.0
      
      #we now create the DAS_control for AP1 or DAS_longControl for AP2
      if self.CP.carFingerprint == CAR.AP2_MODELS:
        can_sends.append(self.tesla_can.create_ap2_long_control(self.v_target, accel_limits_turns, jerk_limits, self.long_control_counter))
      if self.CP.carFingerprint == CAR.AP1_MODELS:
        can_sends.append(self.tesla_can.create_ap1_long_control(self.v_target, test_accel_limits, test_jerk_limits, self.long_control_counter))
      self.long_control_counter = (self.long_control_counter + 1) % 8
     
    if (hands_on_fault):
      enabled = False

    # Cancel when openpilot is not enabled anymore and no autopilot
    # BB: do we need to do this? AP/Tesla does not behave this way
    #   LKAS can be disabled by steering and ACC remains engaged
    if not enabled and bool(CS.out.cruiseState.enabled):
      cruise_cancel = True
      can_sends.append(self.tesla_can.create_action_request(CS.msg_stw_actn_req, cruise_cancel, CANBUS.chassis))
      can_sends.append(self.tesla_can.create_action_request(CS.msg_stw_actn_req, cruise_cancel, CANBUS.autopilot))
      

    # TODO: HUD control: Autopilot Status, (Status2 also needed for long control),
    #       Lanes and BodyControls (keep turn signal on during ALCA)

    return can_sends
