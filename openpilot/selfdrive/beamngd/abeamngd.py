import time
import keyboard
import numpy as np

from openpilot.cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.tools.sim.lib.simulated_car import SimulatedCar
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors
from openpilot.tools.sim.lib.common import SimulatorState
from opendbc.car.honda.values import CruiseButtons

from beamngpy import BeamNGpy, Vehicle
from beamngpy.sensors import Electrics, GForces

BEAMNG_HOST = "localhost"
BEAMNG_PORT = 64256
MAX_STEERING_ANGLE_DEG = 40 # normalize it not a sim issue

class BeamNGBridge:
  def __init__(self):
    self.bng = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT)
    self.bng.open()

    cars = self.bng.get_current_vehicles()
    if not cars:
      raise Exception("No cars.")
      return

    self.id = list(cars.keys())[0]
    car = cars[self.id]

    # car sensors
    self.gforces = GForces()
    self.electrics = Electrics()
    self.car.sensors.attach("electrics", self.electrics)
    self.car.sensors.attach("gforces", self.gforces)
    print("Beamng - Attached to car.")

    # actually start the car
    self.simulated_car = SimulatedCar()
    self.simulated_sensors = SimulatedSensors(dual_camera=True)
    self.simulator_state = SimulatorState()

    # cereal init
    self.sm = messaging.SubMaster(["carControl", "selfDriveState"])

    # ratekeeper (100hz)
    self.rk = Ratekeeper(100, print_delay_threshold=None)

    # states
    self.engaged_prev = False
    self.cruise_button_counter = 0

  def poll_state(self):
    # gets state and telemetry from beamng
    # beamng units are in m/s.
    self.vehicle.sensors.poll()

    # wheelspeeds
    speed_ms = self.electrics.data.get("wheelspeed", 0.0)
    self.simulator_state.velocity = (speed_ms, 0, 0)
    self.simulator_state.valid = True

    # steering
    steering_norm = self.electrics.data.get('steering_input', 0.0)
    self.simulator_state.steering_angle = -steering_norm * MAX_STEERING_ANGLE_DEG

    # user inputs
    self.simulator_state.user_gas = self.electrics.data.get("throttle_input", 0.0)
    self.simulator_state.user_brake = self.electrics.data.get("brake_input", 0.0)
    self.simulator_state.user_torque = -steering_norm * MAX_STEERING_ANGLE_DEG

    # ignition
    self.simulator_state.ignition = True

    # "gee forces"
    gx = self.gforces.data.get('gx2', 0.0)
    gy = self.gforces.data.get('gy2', 0.0)
    gz = self.gforces.data.get('gz2', 0.0)
    self.simulator_state.imu.accelerometer.x = gx * 9.81
    self.simulator_state.imu.accelerometer.y = gy * 9.81
    self.simulator_state.imu.accelerometer.z = gz * 9.81

    # turn signals (no bmws)
    self.simulator_state.left_blinker = bool(self.electrics.data.get('signal_L', 0))
    self.simulator_state.right_blinker = bool(self.electrics.data.get('signal_R', 0))

  def handle_cruise_buttons(self):
    # it well... handles buttons.
    self.sm.update(0)
    self.simulator_state.is_engaged = self.sm['selfdriveState'].active
    self.simulator_state.cruise_button = 0

    # C for DECEL SET. V for RES ACCEL. B for CANCEL.
    # if you dont know what these are check how to use a car's ACC in general
    # or look in the openpilot docs somewhere
    if keyboard.is_pressed('c'):
      self.simulator_state.cruise_button = CruiseButtons.DECEL_SET
    elif keyboard.is_pressed('v'):
      self.simulator_state.cruise_button = CruiseButtons.RES_ACCEL
    elif keyboard.is_pressed('b'):
      self.simulator_state.cruise_button = CruiseButtons.CANCEL

  def apply_openpilot_controls(self):
    if not self.simulator_state.is_engaged: return
    actuators = self.sm['carControl'].actuators
    target_steer_deg = actuators.steeringAngleDeg

    # bring steering from openpilot to the -1 to 1 beamng wants
    steer_out = float(np.clip(-target_steer_deg/MAX_STEERING_ANGLE_DEG, -1.0, 1.0))

    # change accel from the accel rate to actual pedal ins
    accel = actuators.accel
    throttle_out = float(np.clip(accel / 2.0, 0.0, 1.0))
    brake_out = float(np.clip(-accel / 4.0, 0.0, 1.0))

    # send
    self.vehicle.control(steering=steer_out, throttle=throttle_out, brake=brake_out)

  def run(self):
    while True:
      self.poll_beamng_state()
      self.handle_cruise_buttons()
      self.simulated_car.update(self.simulator_state)
      self.simulated_sensors.update(self.simulator_state, None)
      self.apply_openpilot_controls()
      self.rk.keep_time()

def main():
  bridge = BeamNGBridge()
  try:
    bridge.run()
  except KeyboardInterrupt:
    print("beamngd keyboardintterupt, shutting down")

if __name__ == "__main__":
  main()