#!/usr/bin/env python
import gc
import json
import zmq
from cereal import car, log
from common.numpy_fast import clip
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper
from common.profiler import Profiler
from common.params import Params
import selfdrive.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.services import service_list
from selfdrive.car.car_helpers import get_car
from selfdrive.controls.lib.planner import Planner
from selfdrive.controls.lib.drive_helpers import learn_angle_offset, \
                                                 get_events, \
                                                 create_event, \
                                                 EventTypes as ET, \
                                                 update_v_cruise, \
                                                 initialize_v_cruise, \
                                                 kill_defaultd
from selfdrive.controls.lib.longcontrol import LongControl, STARTING_TARGET_SPEED
from selfdrive.controls.lib.latcontrol import LatControl
from selfdrive.controls.lib.alertmanager import AlertManager
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.driver_monitor import DriverStatus

ThermalStatus = log.ThermalData.ThermalStatus
State = log.Live100Data.ControlState


class Calibration:
  UNCALIBRATED = 0
  CALIBRATED = 1
  INVALID = 2


# True when actuators are controlled
def isActive(state):
  return state in [State.enabled, State.softDisabling]


# True if system is engaged
def isEnabled(state):
  return (isActive(state) or state == State.preEnabled)


def data_sample(CI, CC, thermal, calibration, health, driver_monitor, gps_location,
                poller, cal_status, cal_perc, overtemp, free_space, low_battery,
                driver_status, geofence, state, mismatch_counter, params):

  # *** read can and compute car states ***
  CS = CI.update(CC)
  events = list(CS.events)
  enabled = isEnabled(state)

  td = None
  cal = None
  hh = None
  dm = None
  gps = None

  for socket, event in poller.poll(0):
    if socket is thermal:
      td = messaging.recv_one(socket)
    elif socket is calibration:
      cal = messaging.recv_one(socket)
    elif socket is health:
      hh = messaging.recv_one(socket)
    elif socket is driver_monitor:
      dm = messaging.recv_one(socket)
    elif socket is gps_location:
      gps = messaging.recv_one(socket)

  # *** thermal checking logic ***
  # thermal data, checked every second
  if td is not None:
    overtemp = td.thermal.thermalStatus >= ThermalStatus.red

    # under 15% of space free no enable allowed
    free_space = td.thermal.freeSpace < 0.15

    # at zero percent battery, OP should not be allowed
    low_battery = td.thermal.batteryPercent < 1

  if low_battery:
    events.append(create_event('lowBattery', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

  if overtemp:
    events.append(create_event('overheat', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

  if free_space:
    events.append(create_event('outOfSpace', [ET.NO_ENTRY]))

  # *** read calibration status ***
  if cal is not None:
    cal_status = cal.liveCalibration.calStatus
    cal_perc = cal.liveCalibration.calPerc

  if cal_status != Calibration.CALIBRATED:
    if cal_status == Calibration.UNCALIBRATED:
      events.append(create_event('calibrationIncomplete', [ET.NO_ENTRY, ET.SOFT_DISABLE, ET.PERMANENT]))
    else:
      events.append(create_event('calibrationInvalid', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

  if not enabled:
    mismatch_counter = 0

  # *** health checking logic ***
  if hh is not None:
    controls_allowed = hh.health.controlsAllowed
    if not controls_allowed and enabled:
      mismatch_counter += 1

    if mismatch_counter >= 2:
      events.append(create_event('controlsMismatch', [ET.IMMEDIATE_DISABLE]))

  if dm is not None:
    driver_status.get_pose(dm.driverMonitoring, params)

  if geofence is not None and gps is not None:
    geofence.update_geofence_status(gps.gpsLocationExternal, params)

  if geofence is not None and not geofence.in_geofence:
    events.append(create_event('geofence', [ET.NO_ENTRY, ET.WARNING]))

  return CS, events, cal_status, cal_perc, overtemp, free_space, low_battery, mismatch_counter


def calc_plan(CS, CP, events, PL, LaC, LoC, v_cruise_kph, driver_status, geofence):
   # plan runs always, independently of the state
   force_decel = driver_status.awareness < 0. or (geofence is not None and not geofence.in_geofence)
   plan_packet = PL.update(CS, LaC, LoC, v_cruise_kph, force_decel)
   plan = plan_packet.plan
   plan_ts = plan_packet.logMonoTime

   # add events from planner
   events += list(plan.events)

   # disable if lead isn't close when system is active and brake is pressed to avoid
   # unexpected vehicle accelerations
   if CS.brakePressed and plan.vTargetFuture >= STARTING_TARGET_SPEED and not CP.radarOffCan and CS.vEgo < 0.3:
     events.append(create_event('noTarget', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

   return plan, plan_ts


def state_transition(CS, CP, state, events, soft_disable_timer, v_cruise_kph, AM):
  # compute conditional state transitions and execute actions on state transitions
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

  # ***** handle state transitions *****

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
      soft_disable_timer = 300   # 3s TODO: use rate
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


def state_control(plan, CS, CP, state, events, v_cruise_kph, v_cruise_kph_last, AM, rk,
                  driver_status, PL, LaC, LoC, VM, angle_offset, passive, is_metric, cal_perc):
  # Given the state, this function returns the actuators

  # reset actuators to zero
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

  # ***** state specific actions *****

  # DISABLED
  if state in [State.preEnabled, State.disabled]:

    LaC.reset()
    LoC.reset(v_pid=CS.vEgo)

  # ENABLED or SOFT_DISABLING
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

  # *** angle offset learning ***

  if rk.frame % 5 == 2 and plan.lateralValid:
    # *** run this at 20hz again ***
    angle_offset = learn_angle_offset(active, CS.vEgo, angle_offset,
                                      PL.PP.c_poly, PL.PP.c_prob, CS.steeringAngle,
                                      CS.steeringPressed)
  if CS.gasbuttonstatus == 0:
    CP.gasMaxV = [0.2, 0.25, 0.3]
  else:
    CP.gasMaxV = [0.2, 0.5, 0.7]
  # *** gas/brake PID loop ***
  actuators.gas, actuators.brake = LoC.update(active, CS.vEgo, CS.brakePressed, CS.standstill, CS.cruiseState.standstill,
                                              v_cruise_kph, plan.vTarget, plan.vTargetFuture, plan.aTarget,
                                              CP, PL.lead_1)

  # *** steering PID loop ***
  actuators.steer, actuators.steerAngle = LaC.update(active, CS.vEgo, CS.steeringAngle,
                                                     CS.steeringPressed, plan.dPoly, angle_offset, VM, PL,CS.blindspot,CS.leftBlinker,CS.rightBlinker)
 #BB added for ALCA support
  #CS.pid = LaC.pid
  # send a "steering required alert" if saturation count has reached the limit
  if LaC.sat_flag and CP.steerLimitAlert:
    AM.add("steerSaturated", enabled)

  # parse permanent warnings to display constantly
  for e in get_events(events, [ET.PERMANENT]):
    extra_text_1, extra_text_2 = "", ""
    if e == "calibrationIncomplete":
      extra_text_1 = str(cal_perc) + "%"
      extra_text_2 = "35 kph" if is_metric else "15 mph"
    AM.add(str(e) + "Permanent", enabled, extra_text_1=extra_text_1, extra_text_2=extra_text_2)

  # *** process alerts ***
  AM.process_alerts(sec_since_boot())

  return actuators, v_cruise_kph, driver_status, angle_offset


def data_send(perception_state, plan, plan_ts, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, carstate,
              carcontrol, live100, livempc, AM, driver_status,
              LaC, LoC, angle_offset, passive):

  # ***** control the car *****

  CC = car.CarControl.new_message()

  if not passive:

    CC.enabled = isEnabled(state)

    CC.actuators = actuators

    CC.cruiseControl.override = True
    # always cancel if we have an interceptor
    CC.cruiseControl.cancel = not CP.enableCruise or (not isEnabled(state) and CS.cruiseState.enabled)

    # brake discount removes a sharp nonlinearity
    brake_discount = (1.0 - clip(actuators.brake*3., 0.0, 1.0))
    CC.cruiseControl.speedOverride = float(max(0.0, (LoC.v_pid + CS.cruiseState.speedOffset) * brake_discount) if CP.enableCruise else 0.0)
    CC.cruiseControl.accelOverride = CI.calc_accel_override(CS.aEgo, plan.aTarget, CS.vEgo, plan.vTarget)

    CC.hudControl.setSpeed = float(v_cruise_kph * CV.KPH_TO_MS)
    CC.hudControl.speedVisible = isEnabled(state)
    CC.hudControl.lanesVisible = isEnabled(state)
    CC.hudControl.leadVisible = plan.hasLead
    CC.hudControl.visualAlert = AM.visual_alert
    CC.hudControl.audibleAlert = AM.audible_alert

    # send car controls over can
    CI.apply(CC, perception_state)

  # ***** publish state to logger *****
  # publish controls state at 100Hz
  dat = messaging.new_message()
  dat.init('live100')

  dat.live100 = {
    "alertText1": AM.alert_text_1,
    "alertText2": AM.alert_text_2,
    "alertSize": AM.alert_size,
    "alertStatus": AM.alert_status,
    "alertBlinkingRate": AM.alert_rate,
    "alertType": AM.alert_type,
    "awarenessStatus": max(driver_status.awareness, 0.0) if isEnabled(state) else 0.0,
    "driverMonitoringOn": bool(driver_status.monitor_on),
    "canMonoTimes": list(CS.canMonoTimes),
    "planMonoTime": plan_ts,
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
    "vTargetLead": float(plan.vTarget),
    "aTarget": float(plan.aTarget),
    "jerkFactor": float(plan.jerkFactor),
    "angleOffset": float(angle_offset),
    "gpsPlannerActive": plan.gpsPlannerActive,
    "cumLagMs": -rk.remaining*1000.,
  }
  live100.send(dat.to_bytes())

  # broadcast carState
  cs_send = messaging.new_message()
  cs_send.init('carState')
  cs_send.carState = CS
  cs_send.carState.events = events
  carstate.send(cs_send.to_bytes())

  # broadcast carControl
  cc_send = messaging.new_message()
  cc_send.init('carControl')
  cc_send.carControl = CC
  carcontrol.send(cc_send.to_bytes())

  # publish mpc state at 20Hz
  if hasattr(LaC, 'mpc_updated') and LaC.mpc_updated:
    dat = messaging.new_message()
    dat.init('liveMpc')
    dat.liveMpc.x = list(LaC.mpc_solution[0].x)
    dat.liveMpc.y = list(LaC.mpc_solution[0].y)
    dat.liveMpc.psi = list(LaC.mpc_solution[0].psi)
    dat.liveMpc.delta = list(LaC.mpc_solution[0].delta)
    dat.liveMpc.cost = LaC.mpc_solution[0].cost
    livempc.send(dat.to_bytes())

  return CC


def controlsd_thread(gctx=None, rate=100, default_bias=0.):
  gc.disable()

  # start the loop
  set_realtime_priority(3)

  context = zmq.Context()
  params = Params()

  # pub
  live100 = messaging.pub_sock(context, service_list['live100'].port)
  carstate = messaging.pub_sock(context, service_list['carState'].port)
  carcontrol = messaging.pub_sock(context, service_list['carControl'].port)
  livempc = messaging.pub_sock(context, service_list['liveMpc'].port)

  is_metric = params.get("IsMetric") == "1"
  passive = params.get("Passive") != "0"
  if not passive:
    while 1:
      try:
        sendcan = messaging.pub_sock(context, service_list['sendcan'].port)
        break
      except zmq.error.ZMQError:
        kill_defaultd()
  else:
    sendcan = None

  # sub
  poller = zmq.Poller()
  thermal = messaging.sub_sock(context, service_list['thermal'].port, conflate=True, poller=poller)
  health = messaging.sub_sock(context, service_list['health'].port, conflate=True, poller=poller)
  cal = messaging.sub_sock(context, service_list['liveCalibration'].port, conflate=True, poller=poller)
  driver_monitor = messaging.sub_sock(context, service_list['driverMonitoring'].port, conflate=True, poller=poller)
  gps_location = messaging.sub_sock(context, service_list['gpsLocationExternal'].port, conflate=True, poller=poller)

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

  fcw_enabled = params.get("IsFcwEnabled") == "1"
  geofence = None

  PL = Planner(CP, fcw_enabled)
  LoC = LongControl(CP, CI.compute_gb)
  VM = VehicleModel(CP)
  LaC = LatControl(VM)
  AM = AlertManager()
  driver_status = DriverStatus()

  if not passive:
    AM.add("startup", False)

  # write CarParams
  params.put("CarParams", CP.to_bytes())

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

  rk = Ratekeeper(rate, print_delay_threshold=2./1000)

  # learned angle offset
  angle_offset = default_bias
  calibration_params = params.get("CalibrationParams")
  if calibration_params:
    try:
      calibration_params = json.loads(calibration_params)
      angle_offset = calibration_params["angle_offset2"]
    except (ValueError, KeyError):
      pass

  prof = Profiler(False)  # off by default

  while 1:

    prof.checkpoint("Ratekeeper", ignore=True)

    # sample data and compute car events
    CS, events, cal_status, cal_perc, overtemp, free_space, low_battery, mismatch_counter = data_sample(CI, CC, thermal, cal, health,
      driver_monitor, gps_location, poller, cal_status, cal_perc, overtemp, free_space, low_battery, driver_status, geofence, state, mismatch_counter, params)
    prof.checkpoint("Sample")

    # define plan
    plan, plan_ts = calc_plan(CS, CP, events, PL, LaC, LoC, v_cruise_kph, driver_status, geofence)
    prof.checkpoint("Plan")

    if not passive:
      # update control state
      state, soft_disable_timer, v_cruise_kph, v_cruise_kph_last = \
        state_transition(CS, CP, state, events, soft_disable_timer, v_cruise_kph, AM)
      prof.checkpoint("State transition")

    # compute actuators
    actuators, v_cruise_kph, driver_status, angle_offset = state_control(plan, CS, CP, state, events, v_cruise_kph,
      v_cruise_kph_last, AM, rk, driver_status, PL, LaC, LoC, VM, angle_offset, passive, is_metric, cal_perc)
    prof.checkpoint("State Control")

    # publish data
    CC = data_send(PL.perception_state, plan, plan_ts, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, carstate, carcontrol,
      live100, livempc, AM, driver_status, LaC, LoC, angle_offset, passive)
    prof.checkpoint("Sent")

    # *** run loop at fixed rate ***
    rk.keep_time()

    prof.display()


def main(gctx=None):
  controlsd_thread(gctx, 100)

if __name__ == "__main__":
  main()
