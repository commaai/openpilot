#!/usr/bin/env python
import gc
import capnp
from cereal import car, log
from common.numpy_fast import clip
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper, DT_CTRL
from common.profiler import Profiler
from common.params import Params
import selfdrive.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.services import service_list
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.car_helpers import get_car, get_startup_alert
from selfdrive.controls.lib.model_parser import CAMERA_OFFSET
from selfdrive.controls.lib.drive_helpers import get_events, \
                                                 create_event, \
                                                 EventTypes as ET, \
                                                 update_v_cruise, \
                                                 initialize_v_cruise
from selfdrive.controls.lib.longcontrol import LongControl, STARTING_TARGET_SPEED
from selfdrive.controls.lib.latcontrol_pid import LatControlPID
from selfdrive.controls.lib.latcontrol_indi import LatControlINDI
from selfdrive.controls.lib.alertmanager import AlertManager
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.driver_monitor import DriverStatus
from selfdrive.controls.lib.planner import LON_MPC_STEP
from selfdrive.locationd.calibration_helpers import Calibration, Filter

ThermalStatus = log.ThermalData.ThermalStatus
State = log.ControlsState.OpenpilotState


def isActive(state):
  """Check if the actuators are enabled"""
  return state in [State.enabled, State.softDisabling]


def isEnabled(state):
  """Check if openpilot is engaged"""
  return (isActive(state) or state == State.preEnabled)

def events_to_bytes(events):
  # optimization when comparing capnp structs: str() or tree traverse are much slower
  ret = []
  for e in events:
    if isinstance(e, capnp.lib.capnp._DynamicStructReader):
      e = e.as_builder()
    ret.append(e.to_bytes())
  return ret


def data_sample(CI, CC, sm, cal_status, cal_perc, overtemp, free_space, low_battery,
                driver_status, state, mismatch_counter, params):
  """Receive data from sockets and create events for battery, temperature and disk space"""

  # Update carstate from CAN and create events
  CS = CI.update(CC)
  events = list(CS.events)
  enabled = isEnabled(state)

  sm.update(0)

  if sm.updated['thermal']:
    overtemp = sm['thermal'].thermalStatus >= ThermalStatus.red
    free_space = sm['thermal'].freeSpace < 0.07  # under 7% of space free no enable allowed
    low_battery = sm['thermal'].batteryPercent < 1 and sm['thermal'].chargingError  # at zero percent battery, while discharging, OP should not allowed

  # Create events for battery, temperature and disk space
  if low_battery:
    events.append(create_event('lowBattery', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
  if overtemp:
    events.append(create_event('overheat', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
  if free_space:
    events.append(create_event('outOfSpace', [ET.NO_ENTRY]))

  # Handle calibration
  if sm.updated['liveCalibration']:
    cal_status = sm['liveCalibration'].calStatus
    cal_perc = sm['liveCalibration'].calPerc

  if cal_status != Calibration.CALIBRATED:
    if cal_status == Calibration.UNCALIBRATED:
      events.append(create_event('calibrationIncomplete', [ET.NO_ENTRY, ET.SOFT_DISABLE, ET.PERMANENT]))
    else:
      events.append(create_event('calibrationInvalid', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

  # When the panda and controlsd do not agree on controls_allowed
  # we want to disengage openpilot. However the status from the panda goes through
  # another socket other than the CAN messages and one can arrive earlier than the other.
  # Therefore we allow a mismatch for two samples, then we trigger the disengagement.
  if not enabled:
    mismatch_counter = 0

  if sm.updated['health']:
    controls_allowed = sm['health'].controlsAllowed
    if not controls_allowed and enabled:
      mismatch_counter += 1
    if mismatch_counter >= 2:
      events.append(create_event('controlsMismatch', [ET.IMMEDIATE_DISABLE]))

  # Driver monitoring
  if sm.updated['driverMonitoring']:
    driver_status.get_pose(sm['driverMonitoring'], params)

  return CS, events, cal_status, cal_perc, overtemp, free_space, low_battery, mismatch_counter


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

  # DISABLED
  if state == State.disabled:
    if get_events(events, [ET.ENABLE]):
      if get_events(events, [ET.NO_ENTRY]):
        for e in get_events(events, [ET.NO_ENTRY]):
          AM.add(frame, str(e) + "NoEntry", enabled)

      else:
        if get_events(events, [ET.PRE_ENABLE]):
          state = State.preEnabled
        else:
          state = State.enabled
        AM.add(frame, "enable", enabled)
        v_cruise_kph = initialize_v_cruise(CS.vEgo, CS.buttonEvents, v_cruise_kph_last)

  # ENABLED
  elif state == State.enabled:
    if get_events(events, [ET.USER_DISABLE]):
      state = State.disabled
      AM.add(frame, "disable", enabled)

    elif get_events(events, [ET.IMMEDIATE_DISABLE]):
      state = State.disabled
      for e in get_events(events, [ET.IMMEDIATE_DISABLE]):
        AM.add(frame, e, enabled)

    elif get_events(events, [ET.SOFT_DISABLE]):
      state = State.softDisabling
      soft_disable_timer = 300   # 3s
      for e in get_events(events, [ET.SOFT_DISABLE]):
        AM.add(frame, e, enabled)

  # SOFT DISABLING
  elif state == State.softDisabling:
    if get_events(events, [ET.USER_DISABLE]):
      state = State.disabled
      AM.add(frame, "disable", enabled)

    elif get_events(events, [ET.IMMEDIATE_DISABLE]):
      state = State.disabled
      for e in get_events(events, [ET.IMMEDIATE_DISABLE]):
        AM.add(frame, e, enabled)

    elif not get_events(events, [ET.SOFT_DISABLE]):
      # no more soft disabling condition, so go back to ENABLED
      state = State.enabled

    elif get_events(events, [ET.SOFT_DISABLE]) and soft_disable_timer > 0:
      for e in get_events(events, [ET.SOFT_DISABLE]):
        AM.add(frame, e, enabled)

    elif soft_disable_timer <= 0:
      state = State.disabled

  # PRE ENABLING
  elif state == State.preEnabled:
    if get_events(events, [ET.USER_DISABLE]):
      state = State.disabled
      AM.add(frame, "disable", enabled)

    elif get_events(events, [ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE]):
      state = State.disabled
      for e in get_events(events, [ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE]):
        AM.add(frame, e, enabled)

    elif not get_events(events, [ET.PRE_ENABLE]):
      state = State.enabled

  return state, soft_disable_timer, v_cruise_kph, v_cruise_kph_last


def state_control(frame, rcv_frame, plan, path_plan, CS, CP, state, events, v_cruise_kph, v_cruise_kph_last,
                  AM, rk, driver_status, LaC, LoC, VM, read_only, is_metric, cal_perc):
  """Given the state, this function returns an actuators packet"""

  actuators = car.CarControl.Actuators.new_message()

  enabled = isEnabled(state)
  active = isActive(state)

  # check if user has interacted with the car
  driver_engaged = len(CS.buttonEvents) > 0 or \
                   v_cruise_kph != v_cruise_kph_last or \
                   CS.steeringPressed

  # add eventual driver distracted events
  events = driver_status.update(events, driver_engaged, isActive(state), CS.standstill)

  # send FCW alert if triggered by planner
  if plan.fcw:
    AM.add(frame, "fcw", enabled)

  # State specific actions

  if state in [State.preEnabled, State.disabled]:
    LaC.reset()
    LoC.reset(v_pid=CS.vEgo)

  elif state in [State.enabled, State.softDisabling]:
    # parse warnings from car specific interface
    for e in get_events(events, [ET.WARNING]):
      extra_text = ""
      if e == "belowSteerSpeed":
        if is_metric:
          extra_text = str(int(round(CP.minSteerSpeed * CV.MS_TO_KPH))) + " kph"
        else:
          extra_text = str(int(round(CP.minSteerSpeed * CV.MS_TO_MPH))) + " mph"
      AM.add(frame, e, enabled, extra_text_2=extra_text)

  plan_age = DT_CTRL * (frame - rcv_frame['plan'])
  dt = min(plan_age, LON_MPC_STEP + DT_CTRL) + DT_CTRL  # no greater than dt mpc + dt, to prevent too high extraps

  a_acc_sol = plan.aStart + (dt / LON_MPC_STEP) * (plan.aTarget - plan.aStart)
  v_acc_sol = plan.vStart + dt * (a_acc_sol + plan.aStart) / 2.0

  # Gas/Brake PID loop
  actuators.gas, actuators.brake = LoC.update(active, CS.vEgo, CS.brakePressed, CS.standstill, CS.cruiseState.standstill,
                                              v_cruise_kph, v_acc_sol, plan.vTargetFuture, a_acc_sol, CP)
  # Steering PID loop and lateral MPC
  actuators.steer, actuators.steerAngle, lac_log = LaC.update(active, CS.vEgo, CS.steeringAngle, CS.steeringRate,
                                                              CS.steeringPressed, CP, VM, path_plan)

  # Send a "steering required alert" if saturation count has reached the limit
  if LaC.sat_flag and CP.steerLimitAlert:
    AM.add(frame, "steerSaturated", enabled)

  # Parse permanent warnings to display constantly
  for e in get_events(events, [ET.PERMANENT]):
    extra_text_1, extra_text_2 = "", ""
    if e == "calibrationIncomplete":
      extra_text_1 = str(cal_perc) + "%"
      if is_metric:
        extra_text_2 = str(int(round(Filter.MIN_SPEED * CV.MS_TO_KPH))) + " kph"
      else:
        extra_text_2 = str(int(round(Filter.MIN_SPEED * CV.MS_TO_MPH))) + " mph"
    AM.add(frame, str(e) + "Permanent", enabled, extra_text_1=extra_text_1, extra_text_2=extra_text_2)

  AM.process_alerts(frame)

  return actuators, v_cruise_kph, driver_status, v_acc_sol, a_acc_sol, lac_log


def data_send(sm, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, carstate,
              carcontrol, carevents, carparams, controlsstate, sendcan, AM, driver_status,
              LaC, LoC, read_only, start_time, v_acc, a_acc, lac_log, events_prev):
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

  blinker = CS.leftBlinker or CS.rightBlinker
  ldw_allowed = CS.vEgo > 12.5 and not blinker

  if len(list(sm['pathPlan'].rPoly)) == 4:
    CC.hudControl.rightLaneDepart = bool(ldw_allowed and sm['pathPlan'].rPoly[3] > -(1 + CAMERA_OFFSET) and right_lane_visible)
  if len(list(sm['pathPlan'].lPoly)) == 4:
    CC.hudControl.leftLaneDepart = bool(ldw_allowed and sm['pathPlan'].lPoly[3] < (1 - CAMERA_OFFSET) and left_lane_visible)

  CC.hudControl.visualAlert = AM.visual_alert
  CC.hudControl.audibleAlert = AM.audible_alert

  if not read_only:
    # send car controls over can
    can_sends = CI.apply(CC)
    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=CS.canValid))

  force_decel = driver_status.awareness < 0.

  # controlsState
  dat = messaging.new_message()
  dat.init('controlsState')
  dat.valid = CS.canValid
  dat.controlsState = {
    "alertText1": AM.alert_text_1,
    "alertText2": AM.alert_text_2,
    "alertSize": AM.alert_size,
    "alertStatus": AM.alert_status,
    "alertBlinkingRate": AM.alert_rate,
    "alertType": AM.alert_type,
    "alertSound": "",  # no EON sounds yet
    "awarenessStatus": max(driver_status.awareness, 0.0) if isEnabled(state) else 0.0,
    "driverMonitoringOn": bool(driver_status.monitor_on and driver_status.face_detected),
    "canMonoTimes": list(CS.canMonoTimes),
    "planMonoTime": sm.logMonoTime['plan'],
    "pathPlanMonoTime": sm.logMonoTime['pathPlan'],
    "enabled": isEnabled(state),
    "active": isActive(state),
    "vEgo": CS.vEgo,
    "vEgoRaw": CS.vEgoRaw,
    "angleSteers": CS.steeringAngle,
    "curvature": VM.calc_curvature(CS.steeringAngle * CV.DEG_TO_RAD, CS.vEgo),
    "steerOverride": CS.steeringPressed,
    "state": state,
    "engageable": not bool(get_events(events, [ET.NO_ENTRY])),
    "longControlState": LoC.long_control_state,
    "vPid": float(LoC.v_pid),
    "vCruise": float(v_cruise_kph),
    "upAccelCmd": float(LoC.pid.p),
    "uiAccelCmd": float(LoC.pid.i),
    "ufAccelCmd": float(LoC.pid.f),
    "angleSteersDes": float(LaC.angle_steers_des),
    "dampAngleSteersDes": float(LaC.damp_angle_steers_des),
    "vTargetLead": float(v_acc),
    "aTarget": float(a_acc),
    "jerkFactor": float(sm['plan'].jerkFactor),
    "angleModelBias": 0.,
    "gpsPlannerActive": sm['plan'].gpsPlannerActive,
    "vCurvature": sm['plan'].vCurvature,
    "decelForTurn": sm['plan'].decelForTurn,
    "cumLagMs": -rk.remaining * 1000.,
    "startMonoTime": int(start_time * 1e9),
    "mapValid": sm['plan'].mapValid,
    "forceDecel": bool(force_decel),
  }

  if CP.lateralTuning.which() == 'pid':
    dat.controlsState.lateralControlState.pidState = lac_log
  else:
    dat.controlsState.lateralControlState.indiState = lac_log
  controlsstate.send(dat.to_bytes())

  # carState
  cs_send = messaging.new_message()
  cs_send.init('carState')
  cs_send.valid = CS.canValid
  cs_send.carState = CS
  cs_send.carState.events = events
  carstate.send(cs_send.to_bytes())

  # carEvents - logged every second or on change
  events_bytes = events_to_bytes(events)
  if (sm.frame % int(1. / DT_CTRL) == 0) or (events_bytes != events_prev):
    ce_send = messaging.new_message()
    ce_send.init('carEvents', len(events))
    ce_send.carEvents = events
    carevents.send(ce_send.to_bytes())

  # carParams - logged every 50 seconds (> 1 per segment)
  if (sm.frame % int(50. / DT_CTRL) == 0):
    cp_send = messaging.new_message()
    cp_send.init('carParams')
    cp_send.carParams = CP
    carparams.send(cp_send.to_bytes())

  # carControl
  cc_send = messaging.new_message()
  cc_send.init('carControl')
  cc_send.valid = CS.canValid
  cc_send.carControl = CC
  carcontrol.send(cc_send.to_bytes())

  return CC, events_bytes


def controlsd_thread(gctx=None):
  gc.disable()

  # start the loop
  set_realtime_priority(3)

  params = Params()

  # Pub Sockets
  sendcan = messaging.pub_sock(service_list['sendcan'].port)
  controlsstate = messaging.pub_sock(service_list['controlsState'].port)
  carstate = messaging.pub_sock(service_list['carState'].port)
  carcontrol = messaging.pub_sock(service_list['carControl'].port)
  carevents = messaging.pub_sock(service_list['carEvents'].port)
  carparams = messaging.pub_sock(service_list['carParams'].port)

  is_metric = params.get("IsMetric") == "1"
  passive = params.get("Passive") != "0"

  sm = messaging.SubMaster(['thermal', 'health', 'liveCalibration', 'driverMonitoring', 'plan', 'pathPlan'])
  logcan = messaging.sub_sock(service_list['can'].port)

  CC = car.CarControl.new_message()
  CI, CP = get_car(logcan, sendcan)
  AM = AlertManager()

  car_recognized = CP.carName != 'mock'
  # If stock camera is disconnected, we loaded car controls and it's not chffrplus
  controller_available = CP.enableCamera and CI.CC is not None and not passive
  read_only = not car_recognized or not controller_available
  if read_only:
    CP.safetyModel = car.CarParams.SafetyModel.elm327   # diagnostic only

  startup_alert = get_startup_alert(car_recognized, controller_available)
  AM.add(sm.frame, startup_alert, False)

  LoC = LongControl(CP, CI.compute_gb)
  VM = VehicleModel(CP)

  if CP.lateralTuning.which() == 'pid':
    LaC = LatControlPID(CP)
  else:
    LaC = LatControlINDI(CP)

  driver_status = DriverStatus()

  # Write CarParams for radard and boardd safety mode
  params.put("CarParams", CP.to_bytes())
  params.put("LongitudinalControl", "1" if CP.openpilotLongitudinalControl else "0")

  state = State.disabled
  soft_disable_timer = 0
  v_cruise_kph = 255
  v_cruise_kph_last = 0
  overtemp = False
  free_space = False
  cal_status = Calibration.INVALID
  cal_perc = 0
  mismatch_counter = 0
  low_battery = False
  events_prev = []

  sm['pathPlan'].sensorValid = True

  # controlsd is driven by can recv, expected at 100Hz
  rk = Ratekeeper(100, print_delay_threshold=None)

  prof = Profiler(False)  # off by default

  while True:
    start_time = sec_since_boot()
    prof.checkpoint("Ratekeeper", ignore=True)

    # Sample data and compute car events
    CS, events, cal_status, cal_perc, overtemp, free_space, low_battery, mismatch_counter =\
      data_sample(CI, CC, sm, cal_status, cal_perc, overtemp, free_space, low_battery,
                  driver_status, state, mismatch_counter, params)
    prof.checkpoint("Sample")

    # Create alerts
    if not sm.all_alive_and_valid():
      events.append(create_event('commIssue', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not sm['pathPlan'].mpcSolutionValid:
      events.append(create_event('plannerError', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if not sm['pathPlan'].sensorValid:
      events.append(create_event('sensorDataInvalid', [ET.NO_ENTRY, ET.PERMANENT]))
    if not sm['pathPlan'].paramsValid:
      events.append(create_event('vehicleModelInvalid', [ET.WARNING]))
    if not sm['plan'].radarValid:
      events.append(create_event('radarFault', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if sm['plan'].radarCanError:
      events.append(create_event('radarCanError', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not CS.canValid:
      events.append(create_event('canError', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

    # Only allow engagement with brake pressed when stopped behind another stopped car
    if CS.brakePressed and sm['plan'].vTargetFuture >= STARTING_TARGET_SPEED and not CP.radarOffCan and CS.vEgo < 0.3:
      events.append(create_event('noTarget', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

    if not read_only:
      # update control state
      state, soft_disable_timer, v_cruise_kph, v_cruise_kph_last = \
        state_transition(sm.frame, CS, CP, state, events, soft_disable_timer, v_cruise_kph, AM)
      prof.checkpoint("State transition")

    # Compute actuators (runs PID loops and lateral MPC)
    actuators, v_cruise_kph, driver_status, v_acc, a_acc, lac_log = \
      state_control(sm.frame, sm.rcv_frame, sm['plan'], sm['pathPlan'], CS, CP, state, events, v_cruise_kph, v_cruise_kph_last, AM, rk,
                    driver_status, LaC, LoC, VM, read_only, is_metric, cal_perc)

    prof.checkpoint("State Control")

    # Publish data
    CC, events_prev = data_send(sm, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, carstate, carcontrol, carevents, carparams,
                   controlsstate, sendcan, AM, driver_status, LaC, LoC, read_only, start_time, v_acc, a_acc, lac_log, events_prev)
    prof.checkpoint("Sent")

    rk.monitor_time()
    prof.display()


def main(gctx=None):
  controlsd_thread(gctx)


if __name__ == "__main__":
  main()
