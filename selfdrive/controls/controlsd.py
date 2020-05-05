#!/usr/bin/env python3
import os
import gc
from cereal import car, log
from common.numpy_fast import clip
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper, DT_CTRL
from common.profiler import Profiler
from common.params import Params, put_nonblocking
import cereal.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.car_helpers import get_car, get_startup_alert
from selfdrive.controls.lib.lane_planner import CAMERA_OFFSET
from selfdrive.controls.lib.events import Events, ET
from selfdrive.controls.lib.drive_helpers import update_v_cruise, initialize_v_cruise
from selfdrive.controls.lib.longcontrol import LongControl, STARTING_TARGET_SPEED
from selfdrive.controls.lib.latcontrol_pid import LatControlPID
from selfdrive.controls.lib.latcontrol_indi import LatControlINDI
from selfdrive.controls.lib.latcontrol_lqr import LatControlLQR
from selfdrive.controls.lib.alertmanager import AlertManager
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.planner import LON_MPC_STEP
from selfdrive.locationd.calibration_helpers import Calibration, Filter

LANE_DEPARTURE_THRESHOLD = 0.1
STEER_ANGLE_SATURATION_TIMEOUT = 1.0 / DT_CTRL
STEER_ANGLE_SATURATION_THRESHOLD = 2.5  # Degrees

ThermalStatus = log.ThermalData.ThermalStatus
State = log.ControlsState.OpenpilotState
HwType = log.HealthData.HwType

LaneChangeState = log.PathPlan.LaneChangeState
LaneChangeDirection = log.PathPlan.LaneChangeDirection

EventName = car.CarEvent.EventName

def add_lane_change_event(events, path_plan):
  if path_plan.laneChangeState == LaneChangeState.preLaneChange:
    if path_plan.laneChangeDirection == LaneChangeDirection.left:
      events.add(EventName.preLaneChangeLeft)
    else:
      events.add(EventName.preLaneChangeRight)
  elif path_plan.laneChangeState in [LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing]:
      events.add(EventName.laneChange)


def isActive(state):
  """Check if the actuators are enabled"""
  return state in [State.enabled, State.softDisabling]


def isEnabled(state):
  """Check if openpilot is engaged"""
  return (isActive(state) or state == State.preEnabled)


def data_sample(CI, CC, sm, can_sock, state, mismatch_counter, can_error_counter, params):
  """Receive data from sockets and create events for battery, temperature and disk space"""

  # Update carstate from CAN and create events
  can_strs = messaging.drain_sock_raw(can_sock, wait_for_one=True)
  CS = CI.update(CC, can_strs)

  sm.update(0)

  events = Events()
  events.add_from_msg(CS.events)
  events.add_from_msg(sm['dMonitoringState'].events)
  add_lane_change_event(events, sm['pathPlan'])
  enabled = isEnabled(state)

  # Check for CAN timeout
  if not can_strs:
    can_error_counter += 1
    events.add(EventName.canError)

  overtemp = sm['thermal'].thermalStatus >= ThermalStatus.red
  free_space = sm['thermal'].freeSpace < 0.07  # under 7% of space free no enable allowed
  low_battery = sm['thermal'].batteryPercent < 1 and sm['thermal'].chargingError  # at zero percent battery, while discharging, OP should not allowed
  mem_low = sm['thermal'].memUsedPercent > 90

  # Create events for battery, temperature and disk space
  if low_battery:
    events.add(EventName.lowBattery)
  if overtemp:
    events.add(EventName.overheat)
  if free_space:
    events.add(EventName.outOfSpace)
  if mem_low:
    events.add(EventName.lowMemory)

  if CS.stockAeb:
    events.add(EventName.stockAeb)

  # Handle calibration
  cal_status = sm['liveCalibration'].calStatus
  cal_perc = sm['liveCalibration'].calPerc

  if cal_status != Calibration.CALIBRATED:
    if cal_status == Calibration.UNCALIBRATED:
      events.add(EventName.calibrationIncomplete)
    else:
      events.add(EventName.calibrationInvalid)

  if CS.vEgo > 92 * CV.MPH_TO_MS:
    events.add(EventName.speedTooHigh)

  # When the panda and controlsd do not agree on controls_allowed
  # we want to disengage openpilot. However the status from the panda goes through
  # another socket other than the CAN messages and one can arrive earlier than the other.
  # Therefore we allow a mismatch for two samples, then we trigger the disengagement.
  if not enabled:
    mismatch_counter = 0

  controls_allowed = sm['health'].controlsAllowed
  if not controls_allowed and enabled:
    mismatch_counter += 1
  if mismatch_counter >= 200:
    events.add(EventName.controlsMismatch)

  return CS, events, cal_perc, mismatch_counter, can_error_counter


def state_transition(frame, CS, CP, state, events, soft_disable_timer, v_cruise_kph, AM):
  """Compute conditional state transitions and execute actions on state transitions"""
  enabled = isEnabled(state)

  v_cruise_kph_last = v_cruise_kph

  # if stock cruise is completely disabled, then we can use our own set speed logic
  if not CP.enableCruise:
    v_cruise_kph = update_v_cruise(v_cruise_kph, CS.buttonEvents, enabled)
  elif CP.enableCruise and CS.cruiseState.enabled:
    v_cruise_kph = CS.cruiseState.speed * CV.MS_TO_KPH

  # decrease the soft disable timer at every step, as it's reset on
  # entrance in SOFT_DISABLING state
  soft_disable_timer = max(0, soft_disable_timer - 1)

  alert_types = []

  # DISABLED
  if state == State.disabled:
    if events.any([ET.ENABLE]):
      if events.any([ET.NO_ENTRY]):
        alert_types = [ET.NO_ENTRY]

      else:
        if events.any([ET.PRE_ENABLE]):
          state = State.preEnabled
        else:
          state = State.enabled
        alert_types = [ET.ENABLE]
        v_cruise_kph = initialize_v_cruise(CS.vEgo, CS.buttonEvents, v_cruise_kph_last)

  # ENABLED
  elif state == State.enabled:
    if events.any([ET.USER_DISABLE]):
      state = State.disabled
      alert_types = [ET.USER_DISABLE]

    elif events.any([ET.IMMEDIATE_DISABLE]):
      state = State.disabled
      alert_types = [ET.IMMEDIATE_DISABLE]

    elif events.any([ET.SOFT_DISABLE]):
      state = State.softDisabling
      soft_disable_timer = 300   # 3s
      alert_types = [ET.SOFT_DISABLE]

  # SOFT DISABLING
  elif state == State.softDisabling:
    if events.any([ET.USER_DISABLE]):
      state = State.disabled
      alert_types = [ET.USER_DISABLE]

    elif events.any([ET.IMMEDIATE_DISABLE]):
      state = State.disabled
      alert_types = ET.IMMEDIATE_DISABLE

    elif not events.any([ET.SOFT_DISABLE]):
      # no more soft disabling condition, so go back to ENABLED
      state = State.enabled

    elif events.any([ET.SOFT_DISABLE]) and soft_disable_timer > 0:
      alert_types = [ET.SOFT_DISABLE]

    elif soft_disable_timer <= 0:
      state = State.disabled

  # PRE ENABLING
  elif state == State.preEnabled:
    if events.any([ET.USER_DISABLE]):
      state = State.disabled
      alert_types = [ET.USER_DISABLE]

    elif events.any([ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE]):
      state = State.disabled
      alert_types = [ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE]

    elif not events.any([ET.PRE_ENABLE]):
      state = State.enabled

  for t in alert_types:
    for e in events.get_events((t,)):
      AM.add_from_event(frame, e, t, enabled)

  return state, soft_disable_timer, v_cruise_kph, v_cruise_kph_last


def state_control(frame, rcv_frame, plan, path_plan, CS, CP, state, events, v_cruise_kph, v_cruise_kph_last,
                  AM, rk, LaC, LoC, read_only, is_metric, cal_perc, last_blinker_frame, saturated_count):
  """Given the state, this function returns an actuators packet"""

  actuators = car.CarControl.Actuators.new_message()

  enabled = isEnabled(state)
  active = isActive(state)

  if CS.leftBlinker or CS.rightBlinker:
    last_blinker_frame = frame

  if plan.fcw:
    # send FCW alert if triggered by planner
    AM.add(frame, "fcw", enabled)

  elif CS.stockFcw:
    # send a silent alert when stock fcw triggers, since the car is already beeping
    AM.add(frame, "fcwStock", enabled)

  # State specific actions

  if state in [State.preEnabled, State.disabled]:
    LaC.reset()
    LoC.reset(v_pid=CS.vEgo)

  elif state in [State.enabled, State.softDisabling]:
    # parse warnings from car specific interface
    for e in events.get_events([ET.WARNING]):
      extra_text = ""
      if e == "belowSteerSpeed":
        if is_metric:
          extra_text = str(int(round(CP.minSteerSpeed * CV.MS_TO_KPH))) + " kph"
        else:
          extra_text = str(int(round(CP.minSteerSpeed * CV.MS_TO_MPH))) + " mph"
      AM.add_from_event(frame, e, ET.WARNING, enabled, extra_text_2=extra_text)

  plan_age = DT_CTRL * (frame - rcv_frame['plan'])
  dt = min(plan_age, LON_MPC_STEP + DT_CTRL) + DT_CTRL  # no greater than dt mpc + dt, to prevent too high extraps

  a_acc_sol = plan.aStart + (dt / LON_MPC_STEP) * (plan.aTarget - plan.aStart)
  v_acc_sol = plan.vStart + dt * (a_acc_sol + plan.aStart) / 2.0

  # Gas/Brake PID loop
  actuators.gas, actuators.brake = LoC.update(active, CS.vEgo, CS.brakePressed, CS.standstill, CS.cruiseState.standstill,
                                              v_cruise_kph, v_acc_sol, plan.vTargetFuture, a_acc_sol, CP)
  # Steering PID loop and lateral MPC
  actuators.steer, actuators.steerAngle, lac_log = LaC.update(active, CS.vEgo, CS.steeringAngle, CS.steeringRate, CS.steeringTorqueEps, CS.steeringPressed, CS.steeringRateLimited, CP, path_plan)

  # Check for difference between desired angle and angle for angle based control
  angle_control_saturated = CP.steerControlType == car.CarParams.SteerControlType.angle and \
    abs(actuators.steerAngle - CS.steeringAngle) > STEER_ANGLE_SATURATION_THRESHOLD

  saturated_count = saturated_count + 1 if angle_control_saturated and not CS.steeringPressed and active else 0

  # Send a "steering required alert" if saturation count has reached the limit
  if (lac_log.saturated and not CS.steeringPressed) or (saturated_count > STEER_ANGLE_SATURATION_TIMEOUT):
    # Check if we deviated from the path
    left_deviation = actuators.steer > 0 and path_plan.dPoly[3] > 0.1
    right_deviation = actuators.steer < 0 and path_plan.dPoly[3] < -0.1

    if left_deviation or right_deviation:
      AM.add(frame, "steerSaturated", enabled)

  # Parse permanent warnings to display constantly
  for e in events.get_events([ET.PERMANENT]):
    extra_text_1, extra_text_2 = "", ""
    if e == "calibrationIncomplete":
      extra_text_1 = str(cal_perc) + "%"
      if is_metric:
        extra_text_2 = str(int(round(Filter.MIN_SPEED * CV.MS_TO_KPH))) + " kph"
      else:
        extra_text_2 = str(int(round(Filter.MIN_SPEED * CV.MS_TO_MPH))) + " mph"
    AM.add_from_event(frame, e, ET.PERMANENT, enabled, extra_text_1=extra_text_1, extra_text_2=extra_text_2)

  return actuators, v_cruise_kph, v_acc_sol, a_acc_sol, lac_log, last_blinker_frame, saturated_count


def data_send(sm, pm, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, AM,
              LaC, LoC, read_only, start_time, v_acc, a_acc, lac_log, events_prev,
              last_blinker_frame, is_ldw_enabled, can_error_counter):
  """Send actuators and hud commands to the car, send controlsstate and MPC logging"""

  CC = car.CarControl.new_message()
  CC.enabled = isEnabled(state)
  CC.actuators = actuators

  CC.cruiseControl.override = True
  CC.cruiseControl.cancel = not CP.enableCruise or (not isEnabled(state) and CS.cruiseState.enabled)

  # Some override values for Honda
  brake_discount = (1.0 - clip(actuators.brake * 3., 0.0, 1.0))  # brake discount removes a sharp nonlinearity
  CC.cruiseControl.speedOverride = float(max(0.0, (LoC.v_pid + CS.cruiseState.speedOffset) * brake_discount) if CP.enableCruise else 0.0)
  CC.cruiseControl.accelOverride = CI.calc_accel_override(CS.aEgo, sm['plan'].aTarget, CS.vEgo, sm['plan'].vTarget)

  CC.hudControl.setSpeed = float(v_cruise_kph * CV.KPH_TO_MS)
  CC.hudControl.speedVisible = isEnabled(state)
  CC.hudControl.lanesVisible = isEnabled(state)
  CC.hudControl.leadVisible = sm['plan'].hasLead

  right_lane_visible = sm['pathPlan'].rProb > 0.5
  left_lane_visible = sm['pathPlan'].lProb > 0.5
  CC.hudControl.rightLaneVisible = bool(right_lane_visible)
  CC.hudControl.leftLaneVisible = bool(left_lane_visible)

  recent_blinker = (sm.frame - last_blinker_frame) * DT_CTRL < 5.0  # 5s blinker cooldown
  calibrated = sm['liveCalibration'].calStatus == Calibration.CALIBRATED
  ldw_allowed = CS.vEgo > 31 * CV.MPH_TO_MS and not recent_blinker and is_ldw_enabled and not isActive(state) and calibrated

  md = sm['model']
  if len(md.meta.desirePrediction):
    l_lane_change_prob = md.meta.desirePrediction[log.PathPlan.Desire.laneChangeLeft - 1]
    r_lane_change_prob = md.meta.desirePrediction[log.PathPlan.Desire.laneChangeRight - 1]

    l_lane_close = left_lane_visible and (sm['pathPlan'].lPoly[3] < (1.08 - CAMERA_OFFSET))
    r_lane_close = right_lane_visible and (sm['pathPlan'].rPoly[3] > -(1.08 + CAMERA_OFFSET))

    if ldw_allowed:
      CC.hudControl.leftLaneDepart = bool(l_lane_change_prob > LANE_DEPARTURE_THRESHOLD and l_lane_close)
      CC.hudControl.rightLaneDepart = bool(r_lane_change_prob > LANE_DEPARTURE_THRESHOLD and r_lane_close)

  if CC.hudControl.rightLaneDepart or CC.hudControl.leftLaneDepart:
    AM.add(sm.frame, 'ldwPermanent', False)
    events.add(EventName.ldw)

  AM.process_alerts(sm.frame)
  CC.hudControl.visualAlert = AM.visual_alert

  if not read_only:
    # send car controls over can
    can_sends = CI.apply(CC)
    pm.send('sendcan', can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=CS.canValid))

  force_decel = (sm['dMonitoringState'].awarenessStatus < 0.) or (state == State.softDisabling)

  # controlsState
  dat = messaging.new_message('controlsState')
  dat.valid = CS.canValid
  dat.controlsState = {
    "alertText1": AM.alert_text_1,
    "alertText2": AM.alert_text_2,
    "alertSize": AM.alert_size,
    "alertStatus": AM.alert_status,
    "alertBlinkingRate": AM.alert_rate,
    "alertType": AM.alert_type,
    "alertSound": AM.audible_alert,
    "driverMonitoringOn": sm['dMonitoringState'].faceDetected,
    "canMonoTimes": list(CS.canMonoTimes),
    "planMonoTime": sm.logMonoTime['plan'],
    "pathPlanMonoTime": sm.logMonoTime['pathPlan'],
    "enabled": isEnabled(state),
    "active": isActive(state),
    "vEgo": CS.vEgo,
    "vEgoRaw": CS.vEgoRaw,
    "angleSteers": CS.steeringAngle,
    "curvature": VM.calc_curvature((CS.steeringAngle - sm['pathPlan'].angleOffset) * CV.DEG_TO_RAD, CS.vEgo),
    "steerOverride": CS.steeringPressed,
    "state": state,
    "engageable": not bool(events.any([ET.NO_ENTRY])),
    "longControlState": LoC.long_control_state,
    "vPid": float(LoC.v_pid),
    "vCruise": float(v_cruise_kph),
    "upAccelCmd": float(LoC.pid.p),
    "uiAccelCmd": float(LoC.pid.i),
    "ufAccelCmd": float(LoC.pid.f),
    "angleSteersDes": float(LaC.angle_steers_des),
    "vTargetLead": float(v_acc),
    "aTarget": float(a_acc),
    "jerkFactor": float(sm['plan'].jerkFactor),
    "gpsPlannerActive": sm['plan'].gpsPlannerActive,
    "vCurvature": sm['plan'].vCurvature,
    "decelForModel": sm['plan'].longitudinalPlanSource == log.Plan.LongitudinalPlanSource.model,
    "cumLagMs": -rk.remaining * 1000.,
    "startMonoTime": int(start_time * 1e9),
    "mapValid": sm['plan'].mapValid,
    "forceDecel": bool(force_decel),
    "canErrorCounter": can_error_counter,
  }

  if CP.lateralTuning.which() == 'pid':
    dat.controlsState.lateralControlState.pidState = lac_log
  elif CP.lateralTuning.which() == 'lqr':
    dat.controlsState.lateralControlState.lqrState = lac_log
  elif CP.lateralTuning.which() == 'indi':
    dat.controlsState.lateralControlState.indiState = lac_log
  pm.send('controlsState', dat)

  car_events = events.to_msg()

  # carState
  cs_send = messaging.new_message('carState')
  cs_send.valid = CS.canValid
  cs_send.carState = CS
  cs_send.carState.events = car_events
  pm.send('carState', cs_send)

  # carEvents - logged every second or on change
  if (sm.frame % int(1. / DT_CTRL) == 0) or (events.names != events_prev):
    ce_send = messaging.new_message('carEvents', len(events.names))
    ce_send.carEvents = car_events
    pm.send('carEvents', ce_send)

  # carParams - logged every 50 seconds (> 1 per segment)
  if (sm.frame % int(50. / DT_CTRL) == 0):
    cp_send = messaging.new_message('carParams')
    cp_send.carParams = CP
    pm.send('carParams', cp_send)

  # carControl
  cc_send = messaging.new_message('carControl')
  cc_send.valid = CS.canValid
  cc_send.carControl = CC
  pm.send('carControl', cc_send)

  return CC, events.names


def controlsd_thread(sm=None, pm=None, can_sock=None):
  gc.disable()

  # start the loop
  set_realtime_priority(3)

  params = Params()

  is_metric = params.get("IsMetric", encoding='utf8') == "1"
  is_ldw_enabled = params.get("IsLdwEnabled", encoding='utf8') == "1"
  passive = params.get("Passive", encoding='utf8') == "1"
  openpilot_enabled_toggle = params.get("OpenpilotEnabledToggle", encoding='utf8') == "1"
  community_feature_toggle = params.get("CommunityFeaturesToggle", encoding='utf8') == "1"

  passive = passive or not openpilot_enabled_toggle

  # Passive if internet needed
  internet_needed = params.get("Offroad_ConnectivityNeeded", encoding='utf8') is not None
  passive = passive or internet_needed

  # Pub/Sub Sockets
  if pm is None:
    pm = messaging.PubMaster(['sendcan', 'controlsState', 'carState', 'carControl', 'carEvents', 'carParams'])

  if sm is None:
    sm = messaging.SubMaster(['thermal', 'health', 'liveCalibration', 'dMonitoringState', 'plan', 'pathPlan', \
                              'model'])

  if can_sock is None:
    can_timeout = None if os.environ.get('NO_CAN_TIMEOUT', False) else 100
    can_sock = messaging.sub_sock('can', timeout=can_timeout)

  # wait for health and CAN packets
  hw_type = messaging.recv_one(sm.sock['health']).health.hwType
  has_relay = hw_type in [HwType.blackPanda, HwType.uno]
  print("Waiting for CAN messages...")
  messaging.get_one_can(can_sock)

  CI, CP = get_car(can_sock, pm.sock['sendcan'], has_relay)

  car_recognized = CP.carName != 'mock'
  # If stock camera is disconnected, we loaded car controls and it's not chffrplus
  controller_available = CP.enableCamera and CI.CC is not None and not passive
  community_feature_disallowed = CP.communityFeature and not community_feature_toggle
  read_only = not car_recognized or not controller_available or CP.dashcamOnly or community_feature_disallowed
  if read_only:
    CP.safetyModel = car.CarParams.SafetyModel.noOutput

  # Write CarParams for radard and boardd safety mode
  cp_bytes = CP.to_bytes()
  params.put("CarParams", cp_bytes)
  put_nonblocking("CarParamsCache", cp_bytes)
  put_nonblocking("LongitudinalControl", "1" if CP.openpilotLongitudinalControl else "0")

  CC = car.CarControl.new_message()
  AM = AlertManager()

  startup_alert = get_startup_alert(car_recognized, controller_available)
  AM.add(sm.frame, startup_alert, False)

  LoC = LongControl(CP, CI.compute_gb)
  VM = VehicleModel(CP)

  if CP.lateralTuning.which() == 'pid':
    LaC = LatControlPID(CP)
  elif CP.lateralTuning.which() == 'indi':
    LaC = LatControlINDI(CP)
  elif CP.lateralTuning.which() == 'lqr':
    LaC = LatControlLQR(CP)

  state = State.disabled
  soft_disable_timer = 0
  v_cruise_kph = 255
  v_cruise_kph_last = 0
  mismatch_counter = 0
  can_error_counter = 0
  last_blinker_frame = 0
  saturated_count = 0
  events_prev = []

  sm['liveCalibration'].calStatus = Calibration.INVALID
  sm['pathPlan'].sensorValid = True
  sm['pathPlan'].posenetValid = True
  sm['thermal'].freeSpace = 1.
  sm['dMonitoringState'].events = []
  sm['dMonitoringState'].awarenessStatus = 1.
  sm['dMonitoringState'].faceDetected = False

  # detect sound card presence
  sounds_available = not os.path.isfile('/EON') or (os.path.isdir('/proc/asound/card0') and open('/proc/asound/card0/state').read().strip() == 'ONLINE')

  # controlsd is driven by can recv, expected at 100Hz
  rk = Ratekeeper(100, print_delay_threshold=None)


  prof = Profiler(False)  # off by default

  while True:
    start_time = sec_since_boot()
    prof.checkpoint("Ratekeeper", ignore=True)

    # Sample data and compute car events
    CS, events, cal_perc, mismatch_counter, can_error_counter = data_sample(CI, CC, sm, can_sock, state, mismatch_counter, can_error_counter, params)
    prof.checkpoint("Sample")

    # Create alerts
    if not sm.alive['plan'] and sm.alive['pathPlan']:  # only plan not being received: radar not communicating
      events.add(EventName.radarCommIssue)
    elif not sm.all_alive_and_valid():
      events.add(EventName.commIssue)
    if not sm['pathPlan'].mpcSolutionValid:
      events.add(EventName.plannerError)
    if not sm['pathPlan'].sensorValid and os.getenv("NOSENSOR") is None:
      events.add(EventName.sensorDataInvalid)
    if not sm['pathPlan'].paramsValid:
      events.add(EventName.vehicleModelInvalid)
    if not sm['pathPlan'].posenetValid:
      events.add(EventName.posenetInvalid)
    if not sm['plan'].radarValid:
      events.add(EventName.radarFault)
    if sm['plan'].radarCanError:
      events.add(EventName.radarCanError)
    if not CS.canValid:
      events.add(EventName.canError)
    if not sounds_available:
      events.add(EventName.soundsUnavailable)
    if internet_needed:
      events.add(EventName.internetConnectivityNeeded)
    if community_feature_disallowed:
      events.add(EventName.communityFeatureDisallowed)
    if read_only and not passive:
      events.add(EventName.carUnrecognized)
    if log.HealthData.FaultType.relayMalfunction in sm['health'].faults:
      events.add(EventName.relayMalfunction)


    # Only allow engagement with brake pressed when stopped behind another stopped car
    if CS.brakePressed and sm['plan'].vTargetFuture >= STARTING_TARGET_SPEED and not CP.radarOffCan and CS.vEgo < 0.3:
      events.add(EventName.noTarget)

    if not read_only:
      # update control state
      state, soft_disable_timer, v_cruise_kph, v_cruise_kph_last = \
        state_transition(sm.frame, CS, CP, state, events, soft_disable_timer, v_cruise_kph, AM)
      prof.checkpoint("State transition")

    # Compute actuators (runs PID loops and lateral MPC)
    actuators, v_cruise_kph, v_acc, a_acc, lac_log, last_blinker_frame, saturated_count = \
      state_control(sm.frame, sm.rcv_frame, sm['plan'], sm['pathPlan'], CS, CP, state, events, v_cruise_kph, v_cruise_kph_last, AM, rk,
                    LaC, LoC, read_only, is_metric, cal_perc, last_blinker_frame, saturated_count)

    prof.checkpoint("State Control")

    # Publish data
    CC, events_prev = data_send(sm, pm, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, AM, LaC,
                                LoC, read_only, start_time, v_acc, a_acc, lac_log, events_prev, last_blinker_frame,
                                is_ldw_enabled, can_error_counter)
    prof.checkpoint("Sent")

    rk.monitor_time()
    prof.display()


def main(sm=None, pm=None, logcan=None):
  controlsd_thread(sm, pm, logcan)


if __name__ == "__main__":
  main()
