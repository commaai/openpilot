from cereal import car
from common.realtime import DT_CTRL
from common.numpy_fast import interp, clip
from selfdrive.config import Conversions as CV
from selfdrive.car import apply_std_steer_torque_limits, create_gas_command
from selfdrive.car.gm import gmcan
from selfdrive.car.gm.values import DBC, NO_ASCM, CanBus, CarControllerParams
from selfdrive.car.gm.gmcan import create_gas_multiplier_command, create_gas_divisor_command, create_gas_offset_command
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.start_time = 0.
    self.apply_steer_last = 0
    self.lka_steering_cmd_counter_last = -1
    self.lka_icon_status_last = (False, False)
    self.steer_rate_limited = False

    self.params = CarControllerParams()

    #TODO: confirm these values still apply. Based on Bolt Steering Column from 0.7.x
    if CP.carFingerprint in NO_ASCM:
      self.STEER_MAX = 300
      self.STEER_STEP = 1
      self.STEER_DELTA_UP = 3          # ~0.75s time to peak torque (255/50hz/0.75s)
      self.STEER_DELTA_DOWN = 7       # ~0.3s from peak torque to zero
      self.MIN_STEER_SPEED = 3.

    self.packer_pt = CANPacker(DBC[CP.carFingerprint]['pt'])
    self.packer_obj = CANPacker(DBC[CP.carFingerprint]['radar'])
    self.packer_ch = CANPacker(DBC[CP.carFingerprint]['chassis'])

  def update(self, enabled, CS, frame, actuators,
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert):

    P = self.params

    # Send CAN commands.
    can_sends = []

    # Steering (50Hz)
    # Avoid GM EPS faults when transmitting messages too close together: skip this transmit if we just received the
    # next Panda loopback confirmation in the current CS frame.
    if CS.lka_steering_cmd_counter != self.lka_steering_cmd_counter_last:
      self.lka_steering_cmd_counter_last = CS.lka_steering_cmd_counter
    elif (frame % P.STEER_STEP) == 0:
      lkas_enabled = enabled and not (CS.out.steerWarning or CS.out.steerError) and CS.out.vEgo > P.MIN_STEER_SPEED
      if lkas_enabled:
        new_steer = int(round(actuators.steer * P.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
        self.steer_rate_limited = new_steer != apply_steer
      else:
        apply_steer = 0

      self.apply_steer_last = apply_steer
      # GM EPS faults on any gap in received message counters. To handle transient OP/Panda safety sync issues at the
      # moment of disengaging, increment the counter based on the last message known to pass Panda safety checks.
      idx = (CS.lka_steering_cmd_counter + 1) % 4

      can_sends.append(gmcan.create_steering_control(self.packer_pt, CanBus.POWERTRAIN, apply_steer, idx, lkas_enabled))

    if not enabled:
      # Stock ECU sends max regen when not enabled.
      apply_gas = P.MAX_ACC_REGEN
      apply_brake = 0
    else:
      apply_gas = int(round(interp(actuators.accel, P.GAS_LOOKUP_BP, P.GAS_LOOKUP_V)))
      apply_brake = int(round(interp(actuators.accel, P.BRAKE_LOOKUP_BP, P.BRAKE_LOOKUP_V)))

    if CS.CP.carFingerprint not in NO_ASCM:
      # Gas/regen and brakes - all at 25Hz
      if (frame % 4) == 0:
        idx = (frame // 4) % 4

        at_full_stop = enabled and CS.out.standstill
        near_stop = enabled and (CS.out.vEgo < P.NEAR_STOP_BRAKE_PHASE)
        can_sends.append(gmcan.create_friction_brake_command(self.packer_ch, CanBus.CHASSIS, apply_brake, idx, near_stop, at_full_stop))
        can_sends.append(gmcan.create_gas_regen_command(self.packer_pt, CanBus.POWERTRAIN, apply_gas, idx, enabled, at_full_stop))
    elif CS.CP.enableGasInterceptor:
      #It seems in L mode, accel / decel point is around 1/5
      #0----decel-------0.2-------accel----------1
      #new_gas = 0.8 * actuators.gas + 0.2
      #new_brake = 0.2 * actuators.brake
      #I am assuming we should not get both a gas and a break value...
      #final_pedal2 = new_gas - new_brake
      #TODO: Hysteresis
      #TODO: Use friction brake via AEB for harder braking
      # Not sure what the status of the above is - was not being used...


      #JJS - no adjust yet - scaling needs to be -1 <-> +1
      pedal_gas = clip(actuators.accel, 0., 1.)
      #This would be more appropriate?
      #pedal_gas = clip(actuators.gas, 0., 1.)
      if (frame % 4) == 0:
        idx = (frame // 4) % 4
        # send exactly zero if apply_gas is zero. Interceptor will send the max between read value and apply_gas.
        # This prevents unexpected pedal range rescaling
        can_sends.append(create_gas_command(self.packer_pt, pedal_gas, idx))

        # TODO: Test crossflashed brake controller
        # This is only for testing hacked brakes. Assuming command is the same but on PT bus...
        at_full_stop = enabled and CS.out.standstill
        near_stop = enabled and (CS.out.vEgo < P.NEAR_STOP_BRAKE_PHASE)
        can_sends.append(gmcan.create_friction_brake_command(self.packer_pt, CanBus.POWERTRAIN, apply_brake, idx, near_stop, at_full_stop))


        #Only send transform when transform isn't populated
        if (not CS.interceptor_has_transform) and (frame % 8) == 0:
          can_sends.append(create_gas_multiplier_command(self.packer_pt, 1545, idx))
          can_sends.append(create_gas_divisor_command(self.packer_pt, 1000, idx))
          can_sends.append(create_gas_offset_command(self.packer_pt, 25, idx))
          # The ECM in GM vehicles has a different resistance between signal and ground than honda and toyota
          # Since the Pedal's resistance doesn't match, the values read by the ADC are incorrect
          # Output values are fine
          # formula is new_adc = ((raw_adc * MULTIPLIER) / DIVISOR) + OFFSET
          # multiplier and divisor are used because we are limited to 16-bit integers on the panda
          # The read ADC value must be multiplied sufficiently large that the division is integral
          # Note: might be able to do the * 1000 part on pedal, but this is more flexible...
          # Technically these only need to be sent once, but pedal may bounce. Sending on the 8's, probably don't need to be so freq
          # Note: pedal ignores counter for these messages
          # Note: by using the same counter as the actual last gas command, these will be ignored by older pedal firmware

    if CS.CP.carFingerprint not in NO_ASCM:
      # Send dashboard UI commands (ACC status), 25hz
      if (frame % 4) == 0:
        send_fcw = hud_alert == VisualAlert.fcw
        can_sends.append(gmcan.create_acc_dashboard_command(self.packer_pt, CanBus.POWERTRAIN, enabled, hud_v_cruise * CV.MS_TO_KPH, hud_show_car, send_fcw))

      # Radar needs to know current speed and yaw rate (50hz),
      # and that ADAS is alive (10hz)
      time_and_headlights_step = 10
      tt = frame * DT_CTRL

      if frame % time_and_headlights_step == 0:
        idx = (frame // time_and_headlights_step) % 4
        can_sends.append(gmcan.create_adas_time_status(CanBus.OBSTACLE, int((tt - self.start_time) * 60), idx))
        can_sends.append(gmcan.create_adas_headlights_status(self.packer_obj, CanBus.OBSTACLE))

      speed_and_accelerometer_step = 2
      if frame % speed_and_accelerometer_step == 0:
        idx = (frame // speed_and_accelerometer_step) % 4
        can_sends.append(gmcan.create_adas_steering_status(CanBus.OBSTACLE, idx))
        can_sends.append(gmcan.create_adas_accelerometer_speed_status(CanBus.OBSTACLE, CS.out.vEgo, idx))

      if frame % P.ADAS_KEEPALIVE_STEP == 0:
        can_sends += gmcan.create_adas_keepalive(CanBus.POWERTRAIN)

    # Show green icon when LKA torque is applied, and
    # alarming orange icon when approaching torque limit.
    # If not sent again, LKA icon disappears in about 5 seconds.
    # Conveniently, sending camera message periodically also works as a keepalive.
    lka_active = CS.lkas_status == 1
    lka_critical = lka_active and abs(actuators.steer) > 0.9
    lka_icon_status = (lka_active, lka_critical)
    if frame % P.CAMERA_KEEPALIVE_STEP == 0 or lka_icon_status != self.lka_icon_status_last:
      steer_alert = hud_alert in [VisualAlert.steerRequired, VisualAlert.ldw]
      can_sends.append(gmcan.create_lka_icon_command(CanBus.SW_GMLAN, lka_active, lka_critical, steer_alert))
      self.lka_icon_status_last = lka_icon_status

    return can_sends
