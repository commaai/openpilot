#!/usr/bin/env python
import json
from copy import copy
import zmq
from cereal import car, log
from common.numpy_fast import clip
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper
from common.profiler import Profiler
from common.params import Params
import selfdrive.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.services import service_list
from selfdrive.car import get_car
from selfdrive.controls.lib.planner import Planner
from selfdrive.controls.lib.drive_helpers import learn_angle_offset, \
                                                 get_events, \
                                                 create_event, \
                                                 EventTypes as ET, \
                                                 update_v_cruise, \
                                                 initialize_v_cruise
from selfdrive.controls.lib.longcontrol import LongControl, STARTING_TARGET_SPEED
from selfdrive.controls.lib.latcontrol import LatControl
from selfdrive.controls.lib.alertmanager import AlertManager
from selfdrive.controls.lib.vehicle_model import VehicleModel


AWARENESS_TIME = 360.      # 6 minutes limit without user touching steering wheels
AWARENESS_PRE_TIME = 20.   # a first alert is issued 20s before start decelerating the car

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


def data_sample(CI, CC, thermal, calibration, health, poller, cal_status, overtemp, free_space):

  # *** read can and compute car states ***
  CS = CI.update(CC)
  events = list(CS.events)

  td = None
  cal = None
  hh = None

  for socket, event in poller.poll(0):
    if socket is thermal:
      td = messaging.recv_one(socket)
    elif socket is calibration:
      cal = messaging.recv_one(socket)
    elif socket is health:
      hh = messaging.recv_one(socket)

  # *** thermal checking logic ***
  # thermal data, checked every second
  if td is not None:
    # CPU overtemp above 95 deg
    overtemp_proc = any(t > 950 for t in
                        (td.thermal.cpu0, td.thermal.cpu1, td.thermal.cpu2,
                         td.thermal.cpu3, td.thermal.mem, td.thermal.gpu))
    overtemp_bat = td.thermal.bat > 60000 # 60c
    overtemp = overtemp_proc or overtemp_bat

    # under 15% of space free no enable allowed
    free_space = td.thermal.freeSpace < 0.15

  if overtemp:
    events.append(create_event('overheat', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

  if free_space:
    events.append(create_event('outOfSpace', [ET.NO_ENTRY]))

  # *** read calibration status ***
  if cal is not None:
    cal_status = cal.liveCalibration.calStatus

  if cal_status != Calibration.CALIBRATED:
    if cal_status == Calibration.UNCALIBRATED:
      events.append(create_event('calibrationInProgress', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    else:
      events.append(create_event('calibrationInvalid', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

  # *** health checking logic ***
  if hh is not None:
    controls_allowed = hh.health.controlsAllowed
    if not controls_allowed:
      events.append(create_event('controlsMismatch', [ET.IMMEDIATE_DISABLE]))

  return CS, events, cal_status, overtemp, free_space


def calc_plan(CS, events, PL, LaC, LoC, v_cruise_kph, awareness_status):
   # plan runs always, independently of the state
   plan_packet = PL.update(CS, LaC, LoC, v_cruise_kph, awareness_status < -0.)
   plan = plan_packet.plan
   plan_ts = plan_packet.logMonoTime

   # add events from planner
   events += list(plan.events)

   # disable if lead isn't close when system is active and brake is pressed to avoid
   # unexpected vehicle accelerations
   if CS.brakePressed and plan.vTargetFuture >= STARTING_TARGET_SPEED:
     events.append(create_event('noTarget', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

   return plan, plan_ts


def state_transition(CS, CP, state, events, soft_disable_timer, v_cruise_kph, AM):
  # compute conditional state transitions and execute actions on state transitions
  enabled = isEnabled(state)

  v_cruise_kph_last = v_cruise_kph

  # if stock cruise is completely disabled, then we can use our own set speed logic
  if not CP.enableCruise:
    v_cruise_kph = update_v_cruise(v_cruise_kph, CS.buttonEvents, enabled)

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
        v_cruise_kph = initialize_v_cruise(CS.vEgo)

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
                  awareness_status, PL, LaC, LoC, VM, angle_offset, rear_view_allowed, 
                  rear_view_toggle, passive):
  # Given the state, this function returns the actuators

  # reset actuators to zero
  actuators = car.CarControl.Actuators.new_message()

  enabled = isEnabled(state)
  active = isActive(state)

  for b in CS.buttonEvents:
    # button presses for rear view
    if b.type == "leftBlinker" or b.type == "rightBlinker":
      rear_view_toggle = b.pressed and rear_view_allowed

    if (b.type == "altButton1" and b.pressed) and not passive:
      rear_view_toggle = not rear_view_toggle


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

    # decrease awareness status
    awareness_status -= 0.01/(AWARENESS_TIME)
    if awareness_status <= 0.:
      AM.add("driverDistracted", enabled)
    elif awareness_status <= AWARENESS_PRE_TIME / AWARENESS_TIME and \
         awareness_status >= (AWARENESS_PRE_TIME - 4.) / AWARENESS_TIME:
      AM.add("preDriverDistracted", enabled)

    # parse warnings from car specific interface
    for e in get_events(events, [ET.WARNING]):
      AM.add(e, enabled)

  # *** angle offset learning ***

  if rk.frame % 5 == 2 and plan.lateralValid:
    # *** run this at 20hz again ***
    angle_offset = learn_angle_offset(active, CS.vEgo, angle_offset,
                                      PL.PP.c_poly, PL.PP.c_prob, CS.steeringAngle,
                                      CS.steeringPressed)

  # *** gas/brake PID loop ***
  actuators.gas, actuators.brake = LoC.update(active, CS.vEgo, CS.brakePressed, CS.standstill, CS.cruiseState.standstill,
                                              v_cruise_kph, plan.vTarget, plan.vTargetFuture, plan.aTarget,
                                              CP, PL.lead_1)

  # *** steering PID loop ***
  actuators.steer, actuators.steerAngle = LaC.update(active, CS.vEgo, CS.steeringAngle,
                                                     CS.steeringPressed, plan.dPoly, angle_offset, VM, PL)

  if CP.enableCruise and CS.cruiseState.enabled:
    v_cruise_kph = CS.cruiseState.speed * CV.MS_TO_KPH

  # reset conditions for the 6 minutes timout
  if CS.buttonEvents or \
     v_cruise_kph != v_cruise_kph_last or \
     CS.steeringPressed or \
     state in [State.preEnabled, State.disabled]:
    awareness_status = 1.

  # send a "steering required alert" if saturation count has reached the limit
  if LaC.sat_flag and CP.steerLimitAlert:
    AM.add("steerSaturated", enabled)

  # parse permanent warnings to display constantly
  for e in get_events(events, [ET.PERMANENT]):
    AM.add(str(e) + "Permanent", enabled)

  # *** process alerts ***

  AM.process_alerts(sec_since_boot())

  return actuators, v_cruise_kph, awareness_status, angle_offset, rear_view_toggle


def data_send(plan, plan_ts, CS, CI, CP, VM, state, events, actuators, v_cruise_kph, rk, carstate,
              carcontrol, live100, livempc, AM, rear_view_allowed, rear_view_toggle, awareness_status,
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
    CI.apply(CC)

  # ***** publish state to logger *****
  # publish controls state at 100Hz
  dat = messaging.new_message()
  dat.init('live100')

  # show rear view camera on phone if in reverse gear or when button is pressed
  dat.live100.rearViewCam = ('reverseGear' in [e.name for e in events] and rear_view_allowed) or rear_view_toggle
  dat.live100.alertText1 = AM.alert_text_1
  dat.live100.alertText2 = AM.alert_text_2
  dat.live100.alertSize = AM.alert_size
  dat.live100.alertStatus = AM.alert_status
  dat.live100.awarenessStatus = max(awareness_status, 0.0) if isEnabled(state) else 0.0

  # what packets were used to process
  dat.live100.canMonoTimes = list(CS.canMonoTimes)
  dat.live100.planMonoTime = plan_ts

  # if controls is enabled
  dat.live100.enabled = isEnabled(state)
  dat.live100.active = isActive(state)

  # car state
  dat.live100.vEgo = CS.vEgo
  dat.live100.vEgoRaw = CS.vEgoRaw
  dat.live100.angleSteers = CS.steeringAngle
  dat.live100.curvature = VM.calc_curvature(CS.steeringAngle * CV.DEG_TO_RAD, CS.vEgo)
  dat.live100.steerOverride = CS.steeringPressed

  # high level control state
  dat.live100.state = state

  # longitudinal control state
  dat.live100.longControlState = LoC.long_control_state
  dat.live100.vPid = float(LoC.v_pid)
  dat.live100.vCruise = float(v_cruise_kph)
  dat.live100.upAccelCmd = float(LoC.pid.p)
  dat.live100.uiAccelCmd = float(LoC.pid.i)
  dat.live100.ufAccelCmd = float(LoC.pid.f)

  # lateral control state
  dat.live100.angleSteersDes = float(LaC.angle_steers_des)
  dat.live100.upSteer = float(LaC.pid.p)
  dat.live100.uiSteer = float(LaC.pid.i)
  dat.live100.ufSteer = float(LaC.pid.f)

  # processed radar state, should add a_pcm?
  dat.live100.vTargetLead = float(plan.vTarget)
  dat.live100.aTarget = float(plan.aTarget)
  dat.live100.jerkFactor = float(plan.jerkFactor)

  # log learned angle offset
  dat.live100.angleOffset = float(angle_offset)

  # Save GPS planner status
  dat.live100.gpsPlannerActive = plan.gpsPlannerActive

  # lag
  dat.live100.cumLagMs = -rk.remaining*1000.

  live100.send(dat.to_bytes())

  # broadcast carState
  cs_send = messaging.new_message()
  cs_send.init('carState')
  # TODO: override CS.events with all the cumulated events
  cs_send.carState = copy(CS)
  cs_send.carState.events = events
  carstate.send(cs_send.to_bytes())

  # broadcast carControl
  cc_send = messaging.new_message()
  cc_send.init('carControl')
  cc_send.carControl = copy(CC)
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


def controlsd_thread(gctx, rate=100):
  # start the loop
  set_realtime_priority(3)

  context = zmq.Context()

  params = Params()

  # pub
  live100 = messaging.pub_sock(context, service_list['live100'].port)
  carstate = messaging.pub_sock(context, service_list['carState'].port)
  carcontrol = messaging.pub_sock(context, service_list['carControl'].port)
  livempc = messaging.pub_sock(context, service_list['liveMpc'].port)

  passive = params.get("Passive") != "0"
  if not passive:
    sendcan = messaging.pub_sock(context, service_list['sendcan'].port)
  else:
    sendcan = None

  # sub
  poller = zmq.Poller()
  thermal = messaging.sub_sock(context, service_list['thermal'].port, conflate=True, poller=poller)
  health = messaging.sub_sock(context, service_list['health'].port, conflate=True, poller=poller)
  cal = messaging.sub_sock(context, service_list['liveCalibration'].port, conflate=True, poller=poller)

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

  PL = Planner(CP, fcw_enabled)
  LoC = LongControl(CP, CI.compute_gb)
  VM = VehicleModel(CP)
  LaC = LatControl(VM)
  AM = AlertManager()

  if not passive:
    AM.add("startup", False)

  # write CarParams
  params.put("CarParams", CP.to_bytes())

  state = State.disabled
  soft_disable_timer = 0
  v_cruise_kph = 255
  overtemp = False
  free_space = False
  cal_status = Calibration.UNCALIBRATED
  rear_view_toggle = False
  rear_view_allowed = params.get("IsRearViewMirror") == "1"

  # 0.0 - 1.0
  awareness_status = 1.
  v_cruise_kph_last = 0

  rk = Ratekeeper(rate, print_delay_threshold=2./1000)

  # learned angle offset
  angle_offset = 1.5  # Default model bias
  calibration_params = params.get("CalibrationParams")
  if calibration_params:
    try:
      calibration_params = json.loads(calibration_params)
      angle_offset = calibration_params["angle_offset"]
    except (ValueError, KeyError):
      pass

  prof = Profiler(False)  # off by default

  while 1:

    prof.checkpoint("Ratekeeper", ignore=True)  # rk is here

    # sample data and compute car events
    CS, events, cal_status, overtemp, free_space = data_sample(CI, CC, thermal, cal, health, poller, cal_status,
                                                               overtemp, free_space)
    prof.checkpoint("Sample")

    # define plan
    plan, plan_ts = calc_plan(CS, events, PL, LaC, LoC, v_cruise_kph, awareness_status)
    prof.checkpoint("Plan")

    if not passive:
      # update control state
      state, soft_disable_timer, v_cruise_kph, v_cruise_kph_last = state_transition(CS, CP, state, events, soft_disable_timer,
                                                                                    v_cruise_kph, AM)
      prof.checkpoint("State transition")

    # compute actuators
    actuators, v_cruise_kph, awareness_status, angle_offset, rear_view_toggle = state_control(plan, CS, CP, state, events, v_cruise_kph,
                                                                            v_cruise_kph_last, AM, rk, awareness_status, PL, LaC, LoC, VM,
                                                                            angle_offset, rear_view_allowed, rear_view_toggle, passive)
    prof.checkpoint("State Control")

    # publish data
    CC = data_send(plan, plan_ts, CS, CI, CP, VM, state, events, actuators, v_cruise_kph,
                   rk, carstate, carcontrol, live100, livempc, AM, rear_view_allowed,
                   rear_view_toggle, awareness_status, LaC, LoC, angle_offset, passive)
    prof.checkpoint("Sent")

    # *** run loop at fixed rate ***
    rk.keep_time()

    prof.display()


def main(gctx=None):
  controlsd_thread(gctx, 100)

if __name__ == "__main__":
  main()
