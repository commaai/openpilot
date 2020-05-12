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
from selfdrive.controls.lib.drive_helpers import get_events, \
                                                 create_event, \
                                                 EventTypes as ET, \
                                                 update_v_cruise, \
                                                 initialize_v_cruise
from selfdrive.controls.lib.longcontrol import LongControl, STARTING_TARGET_SPEED
from selfdrive.controls.lib.latcontrol_pid import LatControlPID
from selfdrive.controls.lib.latcontrol_indi import LatControlINDI
from selfdrive.controls.lib.latcontrol_lqr import LatControlLQR
from selfdrive.controls.lib.alertmanager import AlertManager
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.planner import LON_MPC_STEP
from selfdrive.locationd.calibration_helpers import Calibration, Filter

LDW_MIN_SPEED = 31 * CV.MPH_TO_MS
LANE_DEPARTURE_THRESHOLD = 0.1
STEER_ANGLE_SATURATION_TIMEOUT = 1.0 / DT_CTRL
STEER_ANGLE_SATURATION_THRESHOLD = 2.5  # Degrees

ThermalStatus = log.ThermalData.ThermalStatus
State = log.ControlsState.OpenpilotState
HwType = log.HealthData.HwType
LongitudinalPlanSource = log.Plan.LongitudinalPlanSource
Desire = log.PathPlan.Desire
LaneChangeState = log.PathPlan.LaneChangeState
LaneChangeDirection = log.PathPlan.LaneChangeDirection


def events_to_bytes(events):
  # optimization when comparing capnp structs: str() or tree traverse are much slower
  ret = []
  for e in events:
    if isinstance(e, capnp.lib.capnp._DynamicStructReader):
      e = e.as_builder()
    if not e.is_root:
      e = e.copy()
    ret.append(e.to_bytes())
  return ret


class Controls:
  def __init__(self, sm=None, pm=None, can_sock=None):
    gc.disable()
    set_realtime_priority(3)

    # Setup sockets
    self.pm = pm
    if self.pm is None:
      self.pm = messaging.PubMaster(['sendcan', 'controlsState', 'carState', \
                                     'carControl', 'carEvents', 'carParams'])

    self.sm = sm
    if self.sm is None:
      self.sm = messaging.SubMaster(['thermal', 'health', 'model', 'liveCalibration', \
                                     'dMonitoringState', 'plan', 'pathPlan'])

    self.can_sock = can_sock
    if can_sock is None:
      can_timeout = None if os.environ.get('NO_CAN_TIMEOUT', False) else 100
      self.can_sock = messaging.sub_sock('can', timeout=can_timeout)

    # wait for one health and one CAN packet
    hw_type = messaging.recv_one(self.sm.sock['health']).health.hwType
    has_relay = hw_type in [HwType.blackPanda, HwType.uno]
    print("Waiting for CAN messages...")
    messaging.get_one_can(self.can_sock)

    self.CI, self.CP = get_car(self.can_sock, self.pm.sock['sendcan'], has_relay)

    # read params
    params = Params()
    self.is_metric = params.get("IsMetric", encoding='utf8') == "1"
    self.is_ldw_enabled = params.get("IsLdwEnabled", encoding='utf8') == "1"
    internet_needed = params.get("Offroad_ConnectivityNeeded", encoding='utf8') is not None
    community_feature_toggle = params.get("CommunityFeaturesToggle", encoding='utf8') == "1"
    openpilot_enabled_toggle = params.get("OpenpilotEnabledToggle", encoding='utf8') == "1"
    passive = params.get("Passive", encoding='utf8') == "1" or \
              internet_needed or not openpilot_enabled_toggle

    car_recognized = self.CP.carName != 'mock'
    # If stock camera is disconnected, we loaded car controls and it's not dashcam mode
    controller_available = self.CP.enableCamera and self.CI.CC is not None and not passive
    community_feature_disallowed = self.CP.communityFeature and not community_feature_toggle
    self.read_only = not car_recognized or not controller_available or \
                       self.CP.dashcamOnly or community_feature_disallowed
    if self.read_only:
      self.CP.safetyModel = car.CarParams.SafetyModel.noOutput

    # Write CarParams for radard and boardd safety mode
    cp_bytes = self.CP.to_bytes()
    params.put("CarParams", cp_bytes)
    put_nonblocking("CarParamsCache", cp_bytes)
    put_nonblocking("LongitudinalControl", "1" if self.CP.openpilotLongitudinalControl else "0")

    self.CC = car.CarControl.new_message()
    self.AM = AlertManager()

    self.LoC = LongControl(self.CP, self.CI.compute_gb)
    self.VM = VehicleModel(self.CP)

    if self.CP.lateralTuning.which() == 'pid':
      self.LaC = LatControlPID(self.CP)
    elif self.CP.lateralTuning.which() == 'indi':
      self.LaC = LatControlINDI(self.CP)
    elif self.CP.lateralTuning.which() == 'lqr':
      self.LaC = LatControlLQR(self.CP)

    self.state = State.disabled
    self.enabled = False
    self.active = False
    self.can_rcv_error = False
    self.soft_disable_timer = 0
    self.v_cruise_kph = 255
    self.v_cruise_kph_last = 0
    self.mismatch_counter = 0
    self.can_error_counter = 0
    self.last_blinker_frame = 0
    self.saturated_count = 0
    self.events_prev = ""

    self.sm['liveCalibration'].calStatus = Calibration.INVALID
    self.sm['pathPlan'].sensorValid = True
    self.sm['pathPlan'].posenetValid = True
    self.sm['thermal'].freeSpace = 1.
    self.sm['dMonitoringState'].events = []
    self.sm['dMonitoringState'].awarenessStatus = 1.
    self.sm['dMonitoringState'].faceDetected = False

    startup_alert = get_startup_alert(car_recognized, controller_available)
    self.AM.add(self.sm.frame, startup_alert, False)

    # controlsd is driven by can recv, expected at 100Hz
    self.rk = Ratekeeper(100, print_delay_threshold=None)

    self.prof = Profiler(False)  # off by default

    # detect sound card presence and ensure successful init
    sounds_available = not os.path.isfile('/EON') or (os.path.isdir('/proc/asound/card0') \
                        and open('/proc/asound/card0/state').read().strip() == 'ONLINE')

    self.static_events = []
    if not sounds_available:
      self.static_events.append(create_event('soundsUnavailable', [ET.NO_ENTRY, ET.PERMANENT]))
    if internet_needed:
      self.static_events.append(create_event('internetConnectivityNeeded', [ET.NO_ENTRY, ET.PERMANENT]))
    if community_feature_disallowed:
      self.static_events.append(create_event('communityFeatureDisallowed', [ET.PERMANENT]))
    if self.read_only and not passive:
      self.static_events.append(create_event('carUnrecognized', [ET.PERMANENT]))


  def create_events(self, CS):
    """Compute carEvents from carState"""

    events = self.static_events.copy()
    events.extend(CS.events)
    events.extend(self.sm['dMonitoringState'].events)

    # Create events for battery, temperature, disk space, and memory
    if self.sm['thermal'].batteryPercent < 1 and self.sm['thermal'].chargingError:
      # at zero percent battery, while discharging, OP should not allowed
      events.append(create_event('lowBattery', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if self.sm['thermal'].thermalStatus >= ThermalStatus.red:
      events.append(create_event('overheat', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if self.sm['thermal'].freeSpace < 0.07:
      # under 7% of space free no enable allowed
      events.append(create_event('outOfSpace', [ET.NO_ENTRY]))
    if self.sm['thermal'].memUsedPercent > 90:
      events.append(create_event('lowMemory', [ET.NO_ENTRY, ET.SOFT_DISABLE, ET.PERMANENT]))

    # Handle calibration status
    cal_status = self.sm['liveCalibration'].calStatus
    if cal_status != Calibration.CALIBRATED:
      if cal_status == Calibration.UNCALIBRATED:
        events.append(create_event('calibrationIncomplete', [ET.NO_ENTRY, ET.SOFT_DISABLE, ET.PERMANENT]))
      else:
        events.append(create_event('calibrationInvalid', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    # Handle lane change
    if self.sm['pathPlan'].laneChangeState == LaneChangeState.preLaneChange:
      if self.sm['path_plan'].laneChangeDirection == LaneChangeDirection.left:
        events.append(create_event('preLaneChangeLeft', [ET.WARNING]))
      else:
        events.append(create_event('preLaneChangeRight', [ET.WARNING]))
    elif self.sm['pathPlan'].laneChangeState in [LaneChangeState.laneChangeStarting, \
                                        LaneChangeState.laneChangeFinishing]:
      events.append(create_event('laneChange', [ET.WARNING]))

    if self.can_rcv_error:
      events.append(create_event('canError', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if self.mismatch_counter >= 200:
      events.append(create_event('controlsMismatch', [ET.IMMEDIATE_DISABLE]))
    if not self.sm.alive['plan'] and self.sm.alive['pathPlan']:
      # only plan not being received: radar not communicating
      events.append(create_event('radarCommIssue', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    elif not self.sm.all_alive_and_valid():
      events.append(create_event('commIssue', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not self.sm['pathPlan'].mpcSolutionValid:
      events.append(create_event('plannerError', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if not self.sm['pathPlan'].sensorValid and os.getenv("NOSENSOR") is None:
      events.append(create_event('sensorDataInvalid', [ET.NO_ENTRY, ET.PERMANENT]))
    if not self.sm['pathPlan'].paramsValid:
      events.append(create_event('vehicleModelInvalid', [ET.WARNING]))
    if not self.sm['pathPlan'].posenetValid:
      events.append(create_event('posenetInvalid', [ET.NO_ENTRY, ET.WARNING]))
    if not self.sm['plan'].radarValid:
      events.append(create_event('radarFault', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if self.sm['plan'].radarCanError:
      events.append(create_event('radarCanError', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not CS.canValid:
      events.append(create_event('canError', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if log.HealthData.FaultType.relayMalfunction in self.sm['health'].faults:
      events.append(create_event('relayMalfunction', [ET.NO_ENTRY, ET.PERMANENT, ET.IMMEDIATE_DISABLE]))

    # Only allow engagement with brake pressed when stopped behind another stopped car
    if CS.brakePressed and self.sm['plan'].vTargetFuture >= STARTING_TARGET_SPEED \
        and not self.CP.radarOffCan and CS.vEgo < 0.3:
      events.append(create_event('noTarget', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))


    # TODO: clean up this alert creation in alerts refactor

    if self.active:
      for e in get_events(events, [ET.WARNING]):
        # TODO: handle non static text in a cleaner way, like a callback
        extra_text = ""
        if e == "belowSteerSpeed":
          if self.is_metric:
            extra_text = str(int(round(self.CP.minSteerSpeed * CV.MS_TO_KPH))) + " kph"
          else:
            extra_text = str(int(round(self.CP.minSteerSpeed * CV.MS_TO_MPH))) + " mph"
        self.AM.add(self.sm.frame, e, self.enabled, extra_text_2=extra_text)

    for e in get_events(events, [ET.PERMANENT]):
      # TODO: handle non static text in a cleaner way, like a callback
      extra_text_1, extra_text_2 = "", ""
      if e == "calibrationIncomplete":
        extra_text_1 = str(self.sm['liveCalibration'].calPerc) + "%"
        if self.is_metric:
          extra_text_2 = str(int(round(Filter.MIN_SPEED * CV.MS_TO_KPH))) + " kph"
        else:
          extra_text_2 = str(int(round(Filter.MIN_SPEED * CV.MS_TO_MPH))) + " mph"
      self.AM.add(self.sm.frame, str(e) + "Permanent", self.enabled, \
                    extra_text_1=extra_text_1, extra_text_2=extra_text_2)

    return events


  def data_sample(self):
    """Receive data from sockets and update carState"""

    # Update carState from CAN
    can_strs = messaging.drain_sock_raw(self.can_sock, wait_for_one=True)
    CS = self.CI.update(self.CC, can_strs)

    self.sm.update(0)

    # Check for CAN timeout
    if not can_strs:
      self.can_error_counter += 1
      self.can_rcv_error = True
    else:
      self.can_rcv_error = False

    # When the panda and controlsd do not agree on controls_allowed
    # we want to disengage openpilot. However the status from the panda goes through
    # another socket other than the CAN messages and one can arrive earlier than the other.
    # Therefore we allow a mismatch for two samples, then we trigger the disengagement.
    if not self.enabled:
      self.mismatch_counter = 0

    if not self.sm['health'].controlsAllowed and self.enabled:
      self.mismatch_counter += 1

    return CS


  def state_transition(self, CS, events):
    """Compute conditional state transitions and execute actions on state transitions"""

    self.v_cruise_kph_last = self.v_cruise_kph

    # if stock cruise is completely disabled, then we can use our own set speed logic
    if not self.CP.enableCruise:
      self.v_cruise_kph = update_v_cruise(self.v_cruise_kph, CS.buttonEvents, self.enabled)
    elif self.CP.enableCruise and CS.cruiseState.enabled:
      self.v_cruise_kph = CS.cruiseState.speed * CV.MS_TO_KPH

    # decrease the soft disable timer at every step, as it's reset on
    # entrance in SOFT_DISABLING state
    self.soft_disable_timer = max(0, self.soft_disable_timer - 1)

    alert_types = []

    # ENABLED, PRE ENABLING, SOFT DISABLING
    if self.state != State.disabled:
      # user and immediate disable always have priority in a non-disabled state
      if get_events(events, [ET.USER_DISABLE]):
        self.state = State.disabled
        self.AM.add(self.sm.frame, "disable", self.enabled)

      elif get_events(events, [ET.IMMEDIATE_DISABLE]):
        self.state = State.disabled
        alert_types = [ET.IMMEDIATE_DISABLE]

      else:
        # ENABLED
        if self.state == State.enabled:
          if get_events(events, [ET.SOFT_DISABLE]):
            self.state = State.softDisabling
            self.soft_disable_timer = 300   # 3s
            alert_types = [ET.SOFT_DISABLE]

        # SOFT DISABLING
        elif self.state == State.softDisabling:
          if not get_events(events, [ET.SOFT_DISABLE]):
            # no more soft disabling condition, so go back to ENABLED
            self.state = State.enabled

          elif get_events(events, [ET.SOFT_DISABLE]) and self.soft_disable_timer > 0:
            alert_types = [ET.SOFT_DISABLE]

          elif self.soft_disable_timer <= 0:
            self.state = State.disabled

        # PRE ENABLING
        elif self.state == State.preEnabled:
          if not get_events(events, [ET.PRE_ENABLE]):
            self.state = State.enabled

    # DISABLED
    elif self.state == State.disabled:
      if get_events(events, [ET.ENABLE]):
        if get_events(events, [ET.NO_ENTRY]):
          for e in get_events(events, [ET.NO_ENTRY]):
            self.AM.add(self.sm.frame, str(e) + "NoEntry", self.enabled)

        else:
          if get_events(events, [ET.PRE_ENABLE]):
            self.state = State.preEnabled
          else:
            self.state = State.enabled
          self.AM.add(self.sm.frame, "enable", self.enabled)
          self.v_cruise_kph = initialize_v_cruise(CS.vEgo, CS.buttonEvents, self.v_cruise_kph_last)

    for e in get_events(events, alert_types):
      self.AM.add(self.sm.frame, e, self.enabled)

    # Check if actuators are enabled
    self.active = self.state == State.enabled or self.state == State.softDisabling

    # Check if openpilot is engaged
    self.enabled = self.active or self.state == State.preEnabled


  def state_control(self, CS, events):
    """Given the state, this function returns an actuators packet"""

    plan = self.sm['plan']
    path_plan = self.sm['pathPlan']

    actuators = car.CarControl.Actuators.new_message()

    if CS.leftBlinker or CS.rightBlinker:
      self.last_blinker_frame = self.sm.frame

    if plan.fcw:
      # send FCW alert if triggered by planner
      self.AM.add(self.sm.frame, "fcw", self.enabled)

    elif CS.stockFcw:
      # send a silent alert when stock fcw triggers, since the car is already beeping
      self.AM.add(self.sm.frame, "fcwStock", self.enabled)

    # State specific actions

    if not self.active:
      self.LaC.reset()
      self.LoC.reset(v_pid=CS.vEgo)

    plan_age = DT_CTRL * (self.sm.frame - self.sm.rcv_frame['plan'])
    # no greater than dt mpc + dt, to prevent too high extraps
    dt = min(plan_age, LON_MPC_STEP + DT_CTRL) + DT_CTRL

    a_acc_sol = plan.aStart + (dt / LON_MPC_STEP) * (plan.aTarget - plan.aStart)
    v_acc_sol = plan.vStart + dt * (a_acc_sol + plan.aStart) / 2.0

    # Gas/Brake PID loop
    actuators.gas, actuators.brake = self.LoC.update(self.active, CS, v_acc_sol, plan.vTargetFuture, a_acc_sol, self.CP)
    # Steering PID loop and lateral MPC
    actuators.steer, actuators.steerAngle, lac_log = self.LaC.update(self.active, CS, self.CP, path_plan)

    # Check for difference between desired angle and angle for angle based control
    angle_control_saturated = self.CP.steerControlType == car.CarParams.SteerControlType.angle and \
      abs(actuators.steerAngle - CS.steeringAngle) > STEER_ANGLE_SATURATION_THRESHOLD

    if angle_control_saturated and not CS.steeringPressed and self.active:
      self.saturated_count += 1

    # Send a "steering required alert" if saturation count has reached the limit
    if (lac_log.saturated and not CS.steeringPressed) or \
        (self.saturated_count > STEER_ANGLE_SATURATION_TIMEOUT):
      # Check if we deviated from the path
      left_deviation = actuators.steer > 0 and path_plan.dPoly[3] > 0.1
      right_deviation = actuators.steer < 0 and path_plan.dPoly[3] < -0.1

      if left_deviation or right_deviation:
        self.AM.add(self.sm.frame, "steerSaturated", self.enabled)

    return actuators, v_acc_sol, a_acc_sol, lac_log


  def publish_logs(self, CS, events, start_time, actuators, v_acc, a_acc, lac_log):
    """Send actuators and hud commands to the car, send controlsstate and MPC logging"""

    CC = car.CarControl.new_message()
    CC.enabled = self.enabled
    CC.actuators = actuators

    CC.cruiseControl.override = True
    CC.cruiseControl.cancel = not self.CP.enableCruise or (not self.enabled and CS.cruiseState.enabled)

    # Some override values for Honda
    # brake discount removes a sharp nonlinearity
    brake_discount = (1.0 - clip(actuators.brake * 3., 0.0, 1.0))
    speed_override = max(0.0, (self.LoC.v_pid + CS.cruiseState.speedOffset) * brake_discount)
    CC.cruiseControl.speedOverride = float(speed_override if self.CP.enableCruise else 0.0)
    CC.cruiseControl.accelOverride = self.CI.calc_accel_override(CS.aEgo, self.sm['plan'].aTarget, CS.vEgo, self.sm['plan'].vTarget)

    CC.hudControl.setSpeed = float(self.v_cruise_kph * CV.KPH_TO_MS)
    CC.hudControl.speedVisible = self.enabled
    CC.hudControl.lanesVisible = self.enabled
    CC.hudControl.leadVisible = self.sm['plan'].hasLead

    right_lane_visible = self.sm['pathPlan'].rProb > 0.5
    left_lane_visible = self.sm['pathPlan'].lProb > 0.5
    CC.hudControl.rightLaneVisible = bool(right_lane_visible)
    CC.hudControl.leftLaneVisible = bool(left_lane_visible)

    recent_blinker = (self.sm.frame - self.last_blinker_frame) * DT_CTRL < 5.0  # 5s blinker cooldown
    ldw_allowed = self.is_ldw_enabled and CS.vEgo > LDW_MIN_SPEED and not recent_blinker \
                    and not self.active and self.sm['liveCalibration'].calStatus == Calibration.CALIBRATED

    meta = self.sm['model'].meta
    if len(meta.desirePrediction) and ldw_allowed:
      l_lane_change_prob = meta.desirePrediction[Desire.laneChangeLeft - 1]
      r_lane_change_prob = meta.desirePrediction[Desire.laneChangeRight - 1]
      l_lane_close = left_lane_visible and (self.sm['pathPlan'].lPoly[3] < (1.08 - CAMERA_OFFSET))
      r_lane_close = right_lane_visible and (self.sm['pathPlan'].rPoly[3] > -(1.08 + CAMERA_OFFSET))

      CC.hudControl.leftLaneDepart = bool(l_lane_change_prob > LANE_DEPARTURE_THRESHOLD and l_lane_close)
      CC.hudControl.rightLaneDepart = bool(r_lane_change_prob > LANE_DEPARTURE_THRESHOLD and r_lane_close)

    if CC.hudControl.rightLaneDepart or CC.hudControl.leftLaneDepart:
      self.AM.add(self.sm.frame, 'ldwPermanent', False)
      events.append(create_event('ldw', [ET.PERMANENT]))

    self.AM.process_alerts(self.sm.frame)
    CC.hudControl.visualAlert = self.AM.visual_alert

    if not self.read_only:
      # send car controls over can
      can_sends = self.CI.apply(CC)
      self.pm.send('sendcan', can_list_to_can_capnp(can_sends, msgtype='sendcan', valid=CS.canValid))

    force_decel = (self.sm['dMonitoringState'].awarenessStatus < 0.) or \
                    (self.state == State.softDisabling)

    steer_angle_rad = (CS.steeringAngle - self.sm['pathPlan'].angleOffset) * CV.DEG_TO_RAD

    # controlsState
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid
    controlsState = dat.controlsState
    controlsState.alertText1 = self.AM.alert_text_1
    controlsState.alertText2 = self.AM.alert_text_2
    controlsState.alertSize = self.AM.alert_size
    controlsState.alertStatus = self.AM.alert_status
    controlsState.alertBlinkingRate = self.AM.alert_rate
    controlsState.alertType = self.AM.alert_type
    controlsState.alertSound = self.AM.audible_alert
    controlsState.driverMonitoringOn = self.sm['dMonitoringState'].faceDetected
    controlsState.canMonoTimes = list(CS.canMonoTimes)
    controlsState.planMonoTime = self.sm.logMonoTime['plan']
    controlsState.pathPlanMonoTime = self.sm.logMonoTime['pathPlan']
    controlsState.enabled = self.enabled
    controlsState.active = self.active
    controlsState.vEgo = CS.vEgo
    controlsState.vEgoRaw = CS.vEgoRaw
    controlsState.angleSteers = CS.steeringAngle
    controlsState.curvature = self.VM.calc_curvature(steer_angle_rad, CS.vEgo)
    controlsState.steerOverride = CS.steeringPressed
    controlsState.state = self.state
    controlsState.engageable = not bool(get_events(events, [ET.NO_ENTRY]))
    controlsState.longControlState = self.LoC.long_control_state
    controlsState.vPid = float(self.LoC.v_pid)
    controlsState.vCruise = float(self.v_cruise_kph)
    controlsState.upAccelCmd = float(self.LoC.pid.p)
    controlsState.uiAccelCmd = float(self.LoC.pid.i)
    controlsState.ufAccelCmd = float(self.LoC.pid.f)
    controlsState.angleSteersDes = float(self.LaC.angle_steers_des)
    controlsState.vTargetLead = float(v_acc)
    controlsState.aTarget = float(a_acc)
    controlsState.jerkFactor = float(self.sm['plan'].jerkFactor)
    controlsState.gpsPlannerActive = self.sm['plan'].gpsPlannerActive
    controlsState.vCurvature = self.sm['plan'].vCurvature
    controlsState.decelForModel = self.sm['plan'].longitudinalPlanSource == LongitudinalPlanSource.model
    controlsState.cumLagMs = -self.rk.remaining * 1000.
    controlsState.startMonoTime = int(start_time * 1e9)
    controlsState.mapValid = self.sm['plan'].mapValid
    controlsState.forceDecel = bool(force_decel)
    controlsState.canErrorCounter = self.can_error_counter

    if self.CP.lateralTuning.which() == 'pid':
      controlsState.lateralControlState.pidState = lac_log
    elif self.CP.lateralTuning.which() == 'lqr':
      controlsState.lateralControlState.lqrState = lac_log
    elif self.CP.lateralTuning.which() == 'indi':
      controlsState.lateralControlState.indiState = lac_log
    self.pm.send('controlsState', dat)

    # carState
    cs_send = messaging.new_message('carState')
    cs_send.valid = CS.canValid
    cs_send.carState = CS
    cs_send.carState.events = events
    self.pm.send('carState', cs_send)

    # carEvents - logged every second or on change
    events_bytes = events_to_bytes(events)
    if (self.sm.frame % int(1. / DT_CTRL) == 0) or (events_bytes != self.events_prev):
      ce_send = messaging.new_message('carEvents', len(events))
      ce_send.carEvents = events
      self.pm.send('carEvents', ce_send)
    self.events_prev = events_bytes

    # carParams - logged every 50 seconds (> 1 per segment)
    if (self.sm.frame % int(50. / DT_CTRL) == 0):
      cp_send = messaging.new_message('carParams')
      cp_send.carParams = self.CP
      self.pm.send('carParams', cp_send)

    # carControl
    cc_send = messaging.new_message('carControl')
    cc_send.valid = CS.canValid
    cc_send.carControl = CC
    self.pm.send('carControl', cc_send)

    # copy CarControl to pass to CarInterface on the next iteration
    self.CC = CC

  def step(self):
    start_time = sec_since_boot()
    self.prof.checkpoint("Ratekeeper", ignore=True)

    # Sample data from sockets and get a carState
    CS = self.data_sample()
    self.prof.checkpoint("Sample")

    events = self.create_events(CS)

    if not self.read_only:
      # Update control state
      self.state_transition(CS, events)
      self.prof.checkpoint("State transition")

    # Compute actuators (runs PID loops and lateral MPC)
    actuators, v_acc, a_acc, lac_log = self.state_control(CS, events)

    self.prof.checkpoint("State Control")

    # Publish data
    self.publish_logs(CS, events, start_time, actuators, v_acc, a_acc, lac_log)
    self.prof.checkpoint("Sent")

  def controlsd_thread(self):
    while True:
      self.step()
      self.rk.monitor_time()
      self.prof.display()

def main(sm=None, pm=None, logcan=None):
  controls = Controls(sm, pm, logcan)
  controls.controlsd_thread()


if __name__ == "__main__":
  main()
