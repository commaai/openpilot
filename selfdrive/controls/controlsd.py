#!/usr/bin/env python3
import os
import gc
import capnp
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
from selfdrive.controls.lib.events import Events, EventTypes as ET
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
  events.add_from_capnp(CS.events)
  events.add_from_capnp(sm['dMonitoringState'].events)
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
      AM.add_from_event(frame, edata_sample(CI, CC, sm, can_sock, state, mismatch_counter, can_error_counter, params)
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
