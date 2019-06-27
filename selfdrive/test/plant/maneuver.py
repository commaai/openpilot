from selfdrive.test.plant.maneuverplots import ManeuverPlot
from selfdrive.test.plant.plant import Plant
import numpy as np


class Maneuver(object):
  def __init__(self, title, duration, **kwargs):
    # Was tempted to make a builder class
    self.distance_lead = kwargs.get("initial_distance_lead", 200.0)
    self.speed = kwargs.get("initial_speed", 0.0)
    self.lead_relevancy = kwargs.get("lead_relevancy", 0)

    self.grade_values = kwargs.get("grade_values", [0.0, 0.0])
    self.grade_breakpoints = kwargs.get("grade_breakpoints", [0.0, duration])
    self.speed_lead_values = kwargs.get("speed_lead_values", [0.0, 0.0])
    self.speed_lead_breakpoints = kwargs.get("speed_lead_breakpoints", [0.0, duration])

    self.cruise_button_presses = kwargs.get("cruise_button_presses", [])

    self.duration = duration
    self.title = title

  def evaluate(self):
    """runs the plant sim and returns (score, run_data)"""
    plant = Plant(
      lead_relevancy = self.lead_relevancy,
      speed = self.speed,
      distance_lead = self.distance_lead
    )

    last_controls_state = None
    plot = ManeuverPlot(self.title)

    buttons_sorted = sorted(self.cruise_button_presses, key=lambda a: a[1])
    current_button = 0
    while plant.current_time() < self.duration:
      while buttons_sorted and plant.current_time() >= buttons_sorted[0][1]:
        current_button = buttons_sorted[0][0]
        buttons_sorted = buttons_sorted[1:]
        print("current button changed to {0}".format(current_button))

      grade = np.interp(plant.current_time(), self.grade_breakpoints, self.grade_values)
      speed_lead = np.interp(plant.current_time(), self.speed_lead_breakpoints, self.speed_lead_values)

      distance, speed, acceleration, distance_lead, brake, gas, steer_torque, fcw, controls_state= plant.step(speed_lead, current_button, grade)
      if controls_state:
        last_controls_state = controls_state[-1]

      d_rel = distance_lead - distance if self.lead_relevancy else 200.
      v_rel = speed_lead - speed if self.lead_relevancy else 0.

      if last_controls_state:
        # print(last_controls_state)
        #develop plots
        plot.add_data(
          time=plant.current_time(),
          gas=gas, brake=brake, steer_torque=steer_torque,
          distance=distance, speed=speed, acceleration=acceleration,
          up_accel_cmd=last_controls_state.upAccelCmd, ui_accel_cmd=last_controls_state.uiAccelCmd,
          uf_accel_cmd=last_controls_state.ufAccelCmd,
          d_rel=d_rel, v_rel=v_rel, v_lead=speed_lead,
          v_target_lead=last_controls_state.vTargetLead, pid_speed=last_controls_state.vPid,
          cruise_speed=last_controls_state.vCruise,
          jerk_factor=last_controls_state.jerkFactor,
          a_target=last_controls_state.aTarget,
          fcw=fcw)

    print("maneuver end")

    return (None, plot)


