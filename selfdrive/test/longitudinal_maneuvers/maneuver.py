from selfdrive.test.longitudinal_maneuvers.plant import Plant
import numpy as np


class Maneuver():
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
    self.checks = kwargs.get("checks", [])

    self.duration = duration
    self.title = title

  def evaluate(self):
    """runs the plant sim and returns (score, run_data)"""
    plant = Plant(
      lead_relevancy=self.lead_relevancy,
      speed=self.speed,
      distance_lead=self.distance_lead
    )

    buttons_sorted = sorted(self.cruise_button_presses, key=lambda a: a[1])
    current_button = 0
    while plant.current_time() < self.duration:
      while buttons_sorted and plant.current_time() >= buttons_sorted[0][1]:
        current_button = buttons_sorted[0][0]
        buttons_sorted = buttons_sorted[1:]
        print("current button changed to {0}".format(current_button))

      grade = np.interp(plant.current_time(), self.grade_breakpoints, self.grade_values)
      speeds_lead = np.interp(plant.current_time() + np.arange(0.,12.,2.), self.speed_lead_breakpoints, self.speed_lead_values)

      # distance, speed, acceleration, distance_lead, brake, gas, steer_torque, fcw, controls_state= plant.step(speed_lead, current_button, grade)
      log = plant.step(speeds_lead, current_button, grade)


      d_rel = log['distance_lead'] - log['distance'] if self.lead_relevancy else 200.
      v_rel = speeds_lead[0] - log['speed'] if self.lead_relevancy else 0.
      log['d_rel'] = d_rel
      log['v_rel'] = v_rel

    valid = True
    if d_rel < 1.0:
      print("Crashed!!!!")
      valid = False

    print("maneuver end", valid)
    return (None, valid)
