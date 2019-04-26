#!/usr/bin/env python
import gc
import zmq
import json
from cereal import car, log
from common.numpy_fast import clip
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper
from common.profiler import Profiler
from common.params import Params
import selfdrive.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.services import service_list
from selfdrive.car.car_helpers import get_car
from selfdrive.controls.lib.drive_helpers import learn_angle_model_bias, \
                                                 get_events, \
                                                 create_event, \
                                                 EventTypes as ET, \
                                                 update_v_cruise, \
                                                 initialize_v_cruise
from selfdrive.controls.lib.longcontrol import LongControl, STARTING_TARGET_SPEED
from selfdrive.controls.lib.latcontrol import LatControl
from selfdrive.controls.lib.alertmanager import AlertManager
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.driver_monitor import DriverStatus
from selfdrive.controls.lib.planner import _DT_MPC
from selfdrive.locationd.calibration_helpers import Calibration, Filter

ThermalStatus = log.ThermalData.ThermalStatus
State = log.Live100Data.ControlState


def isActive(state):
  """Check if the actuators are enabled"""
  return state in [State.enabled, State.softDisabling]


def isEnabled(state):
  """Check if openpilot is engaged"""
  return (isActive(state) or state == State.preEnabled)


def data_sample(CI, CC, plan_sock, path_plan_sock, thermal, calibration, health, driver_monitor,
                poller, cal_status, cal_perc, overtemp, free_space, low_battery,
                driver_status, state, mismatch_counter, params, plan, path_plan):
  """Receive data from sockets and create events for battery, temperature and disk space"""

  # Update carstate from CAN and create events
  CS = CI.update(CC)
  events = list(CS.events)
  enabled = isEnabled(state)

  # Receive from sockets
  td = None
  cal = None
  hh = None
  dm = None

  for socket, event in poller.poll(0):
    if socket is thermal:
      td = messaging.recv_one(socket)
    elif socket is calibration:
      cal = messaging.recv_one(socket)
    elif socket is health:
      hh = messaging.recv_one(socket)
    elif socket is driver_monitor:
      dm = messaging.recv_one(socket)
    elif socket is plan_sock:
      plan = messaging.recv_one(socket)
    elif socket is path_plan_sock:
      path_plan = messaging.recv_one(socket)

  if td is not None:
    overtemp = td.thermal.thermalStatus >= ThermalStatus.red
    free_space = td.thermal.freeSpace < 0.07  # under 7% of space free no enable allowed
    low_battery = td.thermal.batteryPercent < 1 and td.thermal.chargingError  # at zero percent battery, while discharging, OP should not be allowed

  # Create events for battery, temperature and disk space
  if low_battery:
    events.append(create_event('lowBattery', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
  if overtemp:
    events.append(create_event('overheat', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
  if free_space:
    events.append(create_event('outOfSpace', [ET.NO_ENTRY]))

  # Handle calibration
  if cal is not None:
    cal_status = cal.liveCalibration.calStatus
    cal_perc = cal.liveCalibration.calPerc

  if cal_status != Calibration.CALIBRATED:
    if cal_status == Calibration.UNCALIBRATED:
      events.append(create_event('calibrationIncomplete', [ET.NO_ENTRY, ET.SOFT_DISABLE, ET.PERMANENT]))
    else:
      events.append(create_event('calibrationInvalid', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

  # When the panda and controlsd do not agree on controls_allowed
  # we want to disengage openpilot. However the status from the panda goes through
  # another socket than the CAN messages, therefore one can arrive earlier than the other.
  # Therefore we allow a mismatch for two samples, then we trigger the disengagement.
  if not enabled:
    mismatch_counter = 0

  if hh is not None:
    controls_allowed = hh.health.controlsAllowed
    if not controls_allowed and enabled:
      mismatch_counter += 1
    if mismatch_counter >= 2:
      events.append(create_event('controlsMismatch', [ET.IMMEDIATE_DISABLE]))

  # Driver monitoring
  if dm is not None:
    driver_status.get_pose(dm.driverMonitoring, params)

  return CS, events, cal_status, cal_perc, overtemp, free_space, low_battery, mismatch_counter, plan, path_plan


def state_transition(CS, CP, state, events, soft_disable_timer, v_cruise_kph, AM):
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
          AM.add(str(e) + "NoEntry", enabled)

      else:
        if get_events(events, [ET.PRE_ENABLE]):
          state = State.preEnabled
        else:
          state = State.enabled
        AM.add("enable", enabled)
        v_cruise_kph = initialize_v_cruise(CS.vEgo, CS.buttonEvents, v_cruise_kph_last)

  # ENABLED
  elif state == State.enabled:
    if get_events(events, [ET.USER_DISABLE]):
      state = State.disabled
      AM.add("disable", enabled)

    elif get_events(events, [ET.IMMEDIATE_DISABLE]):
      state = State.disabled
      for e in get_events(events, [ET.IMMEDIATE_DISABLE]):
        AM.add(e, enabled)

    elif get_events(events, [ET.SOFT_DISABLE]):
      state = State.softDisabling
      soft_disable_timer = 300   # 3s
      for e in get_events(events, [ET.SOFT_DISABLE]):
        AM.add(e, enabled)

  # SOFT DISABLING
  elif state == State.softDisabling:
    if get_events(events, [ET.USER_DISABLE]):
      state = State.disabled
      AM.add("disable", enabled)

    elif get_events(events, [ET.IMMEDIATE_DISABLE]):
      state = State.disabled
      for e in get_events(events, [ET.IMMEDIATE_DISABLE]):
        AM.add(e, enabled)

    elif not get_events(events, [ET.SOFT_DISABLE]):
      # no more soft disabling condition, so go back to ENABLED
      state = State.enabled

    elif get_events(events, [ET.SOFT_DISABLE]) and soft_disable_timer > 0:
      for e in get_events(events, [ET.SOFT_DISABLE]):
        AM.add(e, enabled)

    elif soft_disable_timer <= 0:
      state = State.disabled

  # PRE ENABLING
  elif state == State.preEnabled:
    if get_events(events, [ET.USER_DISABLE]):
      state = State.disabled
      AM.add("disable", enabled)

    elif get_events(events, [ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE]):
      state = State.disabled
      for e in get_events(events, [ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE]):
        AM.add(e, enabled)

    elif not get_events(events, [ET.PRE_ENABLE]):
      state = State.enabled

  return state, soft_disable_timer, v_cruise_kph, v_cruise_kph_last


def state_control(plan, path_plan, CS, CP, state, events, v_cruise_kph, v_cruise_kph_last, AM, rk,
                  driver_status, LaC, LoC, VM, angle_model_bias, passive, is_metric, cal_perc):
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
    AM.add("fcw", enabled)

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
      AM.add(e, enabled, extra_text_2=extra_text)

  # Run angle offset learner at 20 Hz
  if rk.frame % 5 == 2:
    angle_model_bias = learn_angle_model_bias(active, CS.vEgo, angle_model_bias,
                                      path_plan.cPoly, path_plan.cProb, CS.steeringAngle,
                                      CS.steeringPressed)

  cur_time = sec_since_boot()  # TODO: This won't work in replay
  mpc_time = plan.l20MonoTime / 1e9
  _DT = 0.01 # 100Hz

  dt = min(cur_time - mpc_time, _DT_MPC + _DT) + _DT  # no greater than dt mpc + dt, to prevent too high extraps
  a_acc_sol = plan.aStart + (dt / _DT_MPC) * (plan.aTarget - plan.aStart)
  v_acc_sol = plan.vStart + dt * (a_acc_sol + plan.aStart) / 2.0

  # Gas/Brake PID loop
  actuators.gas, actuators.brake = LoC.update(active, CS.vEgo, CS.brakePressed, CS.standstill, CS.cruiseState.standstill,
                                              v_cruise_kph, v_acc_sol, plan.vTargetFuture, a_acc_sol, CP)
  # Steering PID loop and lateral MPC
  actuators.steer, actuators.steerAngle = LaC.update(active, CS.vEgo, CS.steeringAngle,
                                                     CS.steeringPressed, CP, VM, path_plan)

  # Send a "steering required alert" if saturation count has reached the limit
  if LaC.sat_flag and CP.steerLimitAlert and CS.lkMode and not CS.leftBlinker and not CS.rightBlinker:
    AM.add("steerSaturated", enabled)

  # Parse permanent warnings to display constantly
  for e in get_events(events, [ET.PERMANENT]):
    extra_text_1, extra_text_2 = "", ""
    if e == "calibrationIncomplete":
      extra_text_1 = str(cal_perc) + "%"
      if is_metric:
        extra_text_2 = str(int(round(Filter.MIN_SPEED * CV.MS_TO_KPH))) + " kph"
      else:
        extra_text_2 = str(int(round(Filter.MIN_SPEED * CV.MS_TO_MPH))) + " mph"
    AM.add(str(e) + "Permanent", enabled, extra_text_1=extra_text_1, extra_text_2=extra_text_2)

  AM.process_alerts(sec_since_boot())

  return actuators, v_cruise_kph, driver_status, angle_model_bias, v_acc_sol, a_acc_sol


def data_send(plan, path_plan, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, carstate,
              carcontrol, live100, AM, driver_status,
              LaC, LoC, angle_model_bias, passive, start_time, v_acc, a_acc):
  """Send actuators and hud commands to the car, send live100 and MPC logging"""
  plan_ts = plan.logMonoTime
  plan = plan.plan

  CC = car.CarControl.new_message()

  if not passive:
    CC.enabled = isEnabled(state)
    CC.actuators = actuators

    CC.cruiseControl.override = True
    CC.cruiseControl.cancel = not CP.enableCruise or (not isEnabled(state) and CS.cruiseState.enabled)

    # Some override values for Honda
    brake_discount = (1.0 - clip(actuators.brake * 3., 0.0, 1.0))  # brake discount removes a sharp nonlinearity
    CC.cruiseControl.speedOverride = float(max(0.0, (LoC.v_pid + CS.cruiseState.speedOffset) * brake_discount) if CP.enableCruise else 0.0)
    CC.cruiseControl.accelOverride = CI.calc_accel_override(CS.aEgo, plan.aTarget, CS.vEgo, plan.vTarget)

    CC.hudControl.setSpeed = float(v_cruise_kph * CV.KPH_TO_MS)
    CC.hudControl.speedVisible = isEnabled(state)
    CC.hudControl.lanesVisible = isEnabled(state)
    CC.hudControl.leadVisible = plan.hasLead
    CC.hudControl.rightLaneVisible = bool(path_plan.pathPlan.rProb > 0.5)
    CC.hudControl.leftLaneVisible = bool(path_plan.pathPlan.lProb > 0.5)
    CC.hudControl.visualAlert = AM.visual_alert
    CC.hudControl.audibleAlert = AM.audible_alert

    # send car controls over can
    CI.apply(CC)

  force_decel = driver_status.awareness < 0.

  # live100
  dat = messaging.new_message()
  dat.init('live100')
  dat.live100 = {
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
    "planMonoTime": plan_ts,
    "pathPlanMonoTime": path_plan.logMonoTime,
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
    "upSteer": float(LaC.pid.p),
    "uiSteer": float(LaC.pid.i),
    "ufSteer": float(LaC.pid.f),
    "vTargetLead": float(v_acc),
    "aTarget": float(a_acc),
    "jerkFactor": float(plan.jerkFactor),
    "angleModelBias": float(angle_model_bias),
    "gpsPlannerActive": plan.gpsPlannerActive,
    "vCurvature": plan.vCurvature,
    "decelForTurn": plan.decelForTurn,
    "cumLagMs": -rk.remaining * 1000.,
    "startMonoTime": start_time,
    "mapValid": plan.mapValid,
    "forceDecel": bool(force_decel),
  }
  live100.send(dat.to_bytes())

  # carState
  cs_send = messaging.new_message()
  cs_send.init('carState')
  cs_send.carState = CS
  cs_send.carState.events = events
  carstate.send(cs_send.to_bytes())

  # carControl
  cc_send = messaging.new_message()
  cc_send.init('carControl')
  cc_send.carControl = CC
  carcontrol.send(cc_send.to_bytes())

  return CC


def controlsd_thread(gctx=None, rate=100):
  gc.disable()

  # start the loop
  set_realtime_priority(3)

  context = zmq.Context()
  params = Params()

  # Pub Sockets
  live100 = messaging.pub_sock(context, service_list['live100'].port)
  carstate = messaging.pub_sock(context, service_list['carState'].port)
  carcontrol = messaging.pub_sock(context, service_list['carControl'].port)

  is_metric = params.get("IsMetric") == "1"
  passive = params.get("Passive") != "0"

  # No sendcan if passive
  if not passive:
    sendcan = messaging.pub_sock(context, service_list['sendcan'].port)
  else:
    sendcan = None

  # Sub sockets
  poller = zmq.Poller()
  thermal = messaging.sub_sock(context, service_list['thermal'].port, conflate=True, poller=poller)
  health = messaging.sub_sock(context, service_list['health'].port, conflate=True, poller=poller)
  cal = messaging.sub_sock(context, service_list['liveCalibration'].port, conflate=True, poller=poller)
  driver_monitor = messaging.sub_sock(context, service_list['driverMonitoring'].port, conflate=True, poller=poller)
  plan_sock = messaging.sub_sock(context, service_list['plan'].port, conflate=True, poller=poller)
  path_plan_sock = messaging.sub_sock(context, service_list['pathPlan'].port, conflate=True, poller=poller)
  logcan = messaging.sub_sock(context, service_list['can'].port)

  CC = car.CarControl.new_message()
  CI, CP = get_car(logcan, sendcan, 1.0 if passive else None)

  if CI is None:
    raise Exception("unsupported car")

  # if stock camera is connected, then force passive behavior
  if not CP.enableCamera:
    passive = True
    sendcan = None

  if passive:
    CP.safetyModel = car.CarParams.SafetyModels.noOutput

  LoC = LongControl(CP, CI.compute_gb)
  VM = VehicleModel(CP)
  LaC = LatControl(CP)
  AM = AlertManager()
  driver_status = DriverStatus()

  if not passive:
    AM.add("startup", False)

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

  plan = messaging.new_message()
  plan.init('plan')
  path_plan = messaging.new_message()
  path_plan.init('pathPlan')

  rk = Ratekeeper(rate, print_delay_threshold=2. / 1000)
  controls_params = params.get("ControlsParams")

  # Read angle offset from previous drive
  angle_model_bias = 0.
  if controls_params is not None:
    try:
      controls_params = json.loads(controls_params)
      angle_model_bias = controls_params['angle_model_bias']
    except (ValueError, KeyError):
      pass

  prof = Profiler(False)  # off by default

  while True:
    start_time = int(sec_since_boot() * 1e9)
    prof.checkpoint("Ratekeeper", ignore=True)

    # Sample data and compute car events
    CS, events, cal_status, cal_perc, overtemp, free_space, low_battery, mismatch_counter, plan, path_plan  =\
      data_sample(CI, CC, plan_sock, path_plan_sock, thermal, cal, health, driver_monitor,
                  poller, cal_status, cal_perc, overtemp, free_space, low_battery, driver_status,
                  state, mismatch_counter, params, plan, path_plan)
    prof.checkpoint("Sample")

    path_plan_age = (start_time - path_plan.logMonoTime) / 1e9
    plan_age = (start_time - plan.logMonoTime) / 1e9
    if not path_plan.pathPlan.valid or plan_age > 0.5 or path_plan_age > 0.5:
      events.append(create_event('plannerError', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not path_plan.pathPlan.paramsValid:
      events.append(create_event('vehicleModelInvalid', [ET.WARNING]))
    events += list(plan.plan.events)

    # Only allow engagement with brake pressed when stopped behind another stopped car
    if CS.brakePressed and plan.plan.vTargetFuture >= STARTING_TARGET_SPEED and not CP.radarOffCan and CS.vEgo < 0.3:
      events.append(create_event('noTarget', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

    if not passive:
      # update control state
      state, soft_disable_timer, v_cruise_kph, v_cruise_kph_last = \
        state_transition(CS, CP, state, events, soft_disable_timer, v_cruise_kph, AM)
      prof.checkpoint("State transition")

    # Compute actuators (runs PID loops and lateral MPC)
    actuators, v_cruise_kph, driver_status, angle_model_bias, v_acc, a_acc = \
      state_control(plan.plan, path_plan.pathPlan, CS, CP, state, events, v_cruise_kph,
                    v_cruise_kph_last, AM, rk, driver_status,
                    LaC, LoC, VM, angle_model_bias, passive, is_metric, cal_perc)

    prof.checkpoint("State Control")

    # Publish data
    CC = data_send(plan, path_plan, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, carstate, carcontrol,
                   live100, AM, driver_status, LaC, LoC, angle_model_bias, passive, start_time, v_acc, a_acc)
    prof.checkpoint("Sent")

    rk.keep_time()  # Run at 100Hz
    prof.display()


def main(gctx=None):
  controlsd_thread(gctx, 100)


if __name__ == "__main__":
  main()
