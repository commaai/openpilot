"""
Based on this awesome configuration script by Austin Owens
See his Robodog project on github: https://github.com/AustinOwens/robodog

These configurations are based off of the hoverboard tutorial found on the
Odrive Robotics website located here:
https://docs.odriverobotics.com/hoverboard

@author: Austin Owens (licensed under LGPL-2.1)
"""

import sys
import time
import odrive
# pylint: disable=import-error, no-name-in-module
from odrive.enums import AxisState, ControlMode, EncoderMode
# pylint: disable=import-error, no-name-in-module
from odrive.utils import dump_errors


class HBMotorConfig:
  """
  Class for configuring an Odrive axis for a Hoverboard motor.
  Only works with one Odrive at a time.
  """

  # Use the watchdog timer to stop the motors after 2 seconds when no controll signal is received.
  ENABLE_WATCHDOG = False

  # Hoverboard Kv
  HOVERBOARD_KV = 16.0

  # Min/Max phase inductance of motor
  MIN_PHASE_INDUCTANCE = 0
  MAX_PHASE_INDUCTANCE = 0.001

  # Min/Max phase resistance of motor
  MIN_PHASE_RESISTANCE = 0
  MAX_PHASE_RESISTANCE = 0.5

  # Tolerance for encoder offset float
  ENCODER_OFFSET_FLOAT_TOLERANCE = 0.05

  def _get_odrive(self):
    # pylint: disable=no-member
    return odrive.find_any()

  def _get_axis(self, odrv, axis_num):
    '''
    Gets the correct axis.
    axis_num: Which channel/motor on the odrive your referring to, either 0 or 1.
    '''
    return getattr(odrv, "axis{}".format(axis_num))

  def _save_odrive_config(self, odrv, config_name: str, reboot: bool = False):
    print("Saving {} configuration...".format(config_name))
    odrv.save_configuration()
    print("Configuration saved.")

    if reboot:
      self.reboot_odrive(odrv)

  def reboot_odrive(self, odrv):
    print("Rebooting ODrive...")
    try:
      odrv.reboot()
    # pylint: disable=bare-except
    except Exception as e:
      print("Error rebooting ODrive: {}".format(e))

  def erase_config(self):
    odrv = self._get_odrive()
    print("Erasing pre-exsisting configuration...")
    try:
      odrv.erase_configuration()
    # pylint: disable=bare-except
    except Exception as e:
      print("Error erasing configuration: {}".format(e))

  def configure_odrive(self, axis_0_can_id: int = 0, axis_1_can_id: int = 1):
    """ Configure can bus for odrive """
    odrv = self._get_odrive()

    odrv.can.set_baud_rate(250000)
    odrv.axis0.config.can_node_id = axis_0_can_id
    odrv.axis1.config.can_node_id = axis_1_can_id
    self._save_odrive_config(odrv, "can bus")

  def configure_axis(self, axis_num: int):
    """ Configures the odrive device for a hoverboard motor. """
    odrv = self._get_odrive()
    odrv_axis = self._get_axis(odrv, axis_num)

    # Standard 6.5 inch hoverboard hub motors have 30 permanent magnet
    # poles, and thus 15 pole pairs
    odrv_axis.motor.config.pole_pairs = 15

    # Hoverboard hub motors are quite high resistance compared to the hobby
    # aircraft motors, so we want to use a bit higher voltage for the motor
    # calibration, and set up the current sense gain to be more sensitive.
    # The motors are also fairly high inductance, so we need to reduce the
    # bandwidth of the current controller from the default to keep it
    # stable.
    odrv_axis.motor.config.resistance_calib_max_voltage = 4
    odrv_axis.motor.config.requested_current_range = 25
    odrv_axis.motor.config.current_control_bandwidth = 100

    # Estimated KV but should be measured using the "drill test", which can
    # be found here:
    # https://discourse.odriverobotics.com/t/project-hoverarm/441
    odrv_axis.motor.config.torque_constant = 8.27 / self.HOVERBOARD_KV

    # Hoverboard motors contain hall effect sensors instead of incremental
    # encorders
    odrv_axis.encoder.config.mode = EncoderMode.HALL

    # The hall feedback has 6 states for every pole pair in the motor. Since
    # we have 15 pole pairs, we set the cpr to 15*6 = 90.
    odrv_axis.encoder.config.cpr = 90

    # Since hall sensors are low resolution feedback, we also bump up the
    # offset calibration displacement to get better calibration accuracy.
    odrv_axis.encoder.config.calib_scan_distance = 150

    # Since the hall feedback only has 90 counts per revolution, we want to
    # reduce the velocity tracking bandwidth to get smoother velocity
    # estimates. We can also set these fairly modest gains that will be a
    # bit sloppy but shouldn’t shake your rig apart if it’s built poorly.
    # Make sure to tune the gains up when you have everything else working
    # to a stiffness that is applicable to your application.
    odrv_axis.encoder.config.bandwidth = 100
    odrv_axis.controller.config.pos_gain = 1
    odrv_axis.controller.config.vel_gain = 0.02 * \
        odrv_axis.motor.config.torque_constant * odrv_axis.encoder.config.cpr
    odrv_axis.controller.config.vel_integrator_gain = 0.1 * \
        odrv_axis.motor.config.torque_constant * odrv_axis.encoder.config.cpr
    # odrv_axis.controller.config.vel_limit = 10

    # Set to torque control mode so we can directly controll the torque from openpilot
    odrv_axis.controller.config.control_mode = ControlMode.TORQUE_CONTROL

    # Automatic startup into closed loop controll mode
    odrv_axis.config.startup_closed_loop_control = True

    # Savety watchdog
    # Enable watchdog timer to stop the motors if no controll command is
    # recieved for 2 seconds
    odrv_axis.config.enable_watchdog = self.ENABLE_WATCHDOG
    odrv_axis.config.watchdog_timeout = 2.0

    # Saverty velocity & current limits
    # set current limit to 5A and velocity limit to 2 turns per second
    odrv_axis.controller.config.enable_current_mode_vel_limit = True
    odrv_axis.motor.config.current_lim = 10
    odrv_axis.controller.config.vel_limit = 4

    # In the next step we are going to start powering the motor and so we
    # want to make sure that some of the above settings that require a
    # reboot are applied first.
    self._save_odrive_config(odrv, "motor", reboot=True)

  def calibrate(self, axis_num: int):
    """ Calibrates the axis. """

    odrv = self._get_odrive()
    odrv_axis = self._get_axis(odrv, axis_num)

    input("Make sure the motor is free to move, then press enter...")

    print("Calibrating Odrive for hoverboard motor (you should hear a beep)...")

    odrv_axis.requested_state = AxisState.MOTOR_CALIBRATION

    # Wait for calibration to take place
    time.sleep(10)

    if odrv_axis.motor.error != 0:
      print(("Error: Odrive reported an error of {} while in the state " +
            "AXIS_STATE_MOTOR_CALIBRATION. Printing out Odrive motor data for " +
             "debug:\n{}").format(odrv_axis.motor.error, odrv_axis.motor))

      sys.exit(1)

    if odrv_axis.motor.config.phase_inductance <= self.MIN_PHASE_INDUCTANCE or \
            odrv_axis.motor.config.phase_inductance >= self.MAX_PHASE_INDUCTANCE:
      print(("Error: After odrive motor calibration, the phase inductance " +
            "is at {}, which is outside of the expected range. Either widen the " +
             "boundaries of MIN_PHASE_INDUCTANCE and MAX_PHASE_INDUCTANCE (which " +
             "is between {} and {} respectively) or debug/fix your setup. Printing " +
             "out Odrive motor data for debug:\n{}"
             ).format(odrv_axis.motor.config.phase_inductance, self.MIN_PHASE_INDUCTANCE,
                      self.MAX_PHASE_INDUCTANCE, odrv_axis.motor))

      sys.exit(1)

    if odrv_axis.motor.config.phase_resistance <= self.MIN_PHASE_RESISTANCE or \
            odrv_axis.motor.config.phase_resistance >= self.MAX_PHASE_RESISTANCE:
      print(("Error: After odrive motor calibration, the phase resistance " +
            "is at {}, which is outside of the expected range. Either raise the " +
             "MAX_PHASE_RESISTANCE (which is between {} and {} respectively) or " +
             "debug/fix your setup. Printing out Odrive motor data for " +
             "debug:\n{}"
             ).format(odrv_axis.motor.config.phase_resistance, self.MIN_PHASE_RESISTANCE,
                      self.MAX_PHASE_RESISTANCE, odrv_axis.motor))

      sys.exit(1)

    # If all looks good, then lets tell ODrive that saving this calibration
    # to persistent memory is OK
    odrv_axis.motor.config.pre_calibrated = True

    # Check the alignment between the motor and the hall sensor. Because of
    # this step you are allowed to plug the motor phases in random order and
    # also the hall signals can be random. Just don’t change it after
    # calibration.
    print("Calibrating Odrive for encoder...")
    odrv_axis.requested_state = AxisState.ENCODER_OFFSET_CALIBRATION

    # Wait for calibration to take place
    time.sleep(30)

    if odrv_axis.encoder.error != 0:
      print(("Error: Odrive reported an error of {} while in the state " +
            "AXIS_STATE_ENCODER_OFFSET_CALIBRATION. Printing out Odrive encoder " +
             "data for debug:\n{}"
             ).format(odrv_axis.encoder.error,                             odrv_axis.encoder))
      sys.exit(1)

    # If offset_float isn't 0.5 within some tolerance, or its not 1.5 within
    # some tolerance, raise an error
    if not ((odrv_axis.encoder.config.offset_float > 0.5 - self.ENCODER_OFFSET_FLOAT_TOLERANCE and
             odrv_axis.encoder.config.offset_float < 0.5 + self.ENCODER_OFFSET_FLOAT_TOLERANCE) or
            (odrv_axis.encoder.config.offset_float > 1.5 - self.ENCODER_OFFSET_FLOAT_TOLERANCE and
             odrv_axis.encoder.config.offset_float < 1.5 + self.ENCODER_OFFSET_FLOAT_TOLERANCE)):
      print(("Error: After odrive encoder calibration, the 'offset_float' " +
            "is at {}, which is outside of the expected range. 'offset_float' " +
             "should be close to 0.5 or 1.5 with a tolerance of {}. Either " +
             "increase the tolerance or debug/fix your setup. Printing out " +
             "Odrive encoder data for debug:\n{}"
             ).format(odrv_axis.encoder.config.offset_float, self.ENCODER_OFFSET_FLOAT_TOLERANCE, odrv_axis.encoder))
      sys.exit(1)

    # If all looks good, then lets tell ODrive that saving this calibration
    # to persistent memory is OK
    odrv_axis.encoder.config.pre_calibrated = True

    self._save_odrive_config(odrv, "calibration", reboot=True)

  def test_axis_torque(self, axis_num: int):
    """ Tests the axis. """

    odrv = self._get_odrive()
    odrv_axis = self._get_axis(odrv, axis_num)

    print("Motor test")

    # Go from 0 to 360 degrees in increments of 30 degrees
    for i in range(0, 10):
      torque_Nm = i * 0.2
      print("Setting motor torque to {} Nm.".format(torque_Nm))
      odrv_axis.controller.input_torque = torque_Nm
      time.sleep(5)

    print("Placing motor in idle. If you move motor, motor will move freely")
    odrv_axis.requested_state = AxisState.IDLE

  def log_errors(self, odrv):
    """ Dump odrive errors to console"""
    odrv = self._get_odrive()
    dump_errors(odrv)


def main():
  odrive_config = HBMotorConfig()

  # If the programm can't find the odrive after reboot, reconnect the odrive or restart the programm

  # odrive_config.erase_config()
  odrive_config.configure_odrive()

  # Do the following steps for axis 0 and 1
  axis_num = 0

  odrive_config.configure_axis(axis_num)
  # odrive_config.calibrate(axis_num)
  odrive_config.test_axis_torque(axis_num)

  print("Finished")


if __name__ == "__main__":
  main()
