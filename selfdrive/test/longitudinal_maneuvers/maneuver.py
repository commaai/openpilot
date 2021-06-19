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

    valid = True
    current_button = 0
    while plant.current_time() < self.duration:
      grade = np.interp(plant.current_time(), self.grade_breakpoints, self.grade_values)
      speed_lead = np.interp(plant.current_time(), self.speed_lead_breakpoints, self.speed_lead_values)

      # distance, speed, acceleration, distance_lead, brake, gas, steer_torque, fcw, controls_state= plant.step(speed_lead, current_button, grade)
      log = plant.step(speed_lead, current_button, grade)


      d_rel = log['distance_lead'] - log['distance'] if self.lead_relevancy else 200.
      v_rel = speed_lead - log['speed'] if self.lead_relevancy else 0.
      log['d_rel'] = d_rel
      log['v_rel'] = v_rel

    if d_rel < 1.0:
      print("Crashed!!!!")
      valid = False

    print("maneuver end", valid)
    return (None, valid)
