import numpy as np
from selfdrive.test.longitudinal_maneuvers.plant import Plant


class Maneuver():
  def __init__(self, title, duration, **kwargs):
    # Was tempted to make a builder class
    self.distance_lead = kwargs.get("initial_distance_lead", 200.0)
    self.speed = kwargs.get("initial_speed", 0.0)
    self.lead_relevancy = kwargs.get("lead_relevancy", 0)

    self.speed_lead_values = kwargs.get("speed_lead_values", [0.0, 0.0])
    self.speed_lead_breakpoints = kwargs.get("speed_lead_breakpoints", [0.0, duration])

    self.only_lead2 = kwargs.get("only_lead2", False)
    self.only_radar = kwargs.get("only_radar", False)

    self.duration = duration
    self.title = title

  def evaluate(self):
    plant = Plant(
      lead_relevancy=self.lead_relevancy,
      speed=self.speed,
      distance_lead=self.distance_lead,
      only_lead2=self.only_lead2,
      only_radar=self.only_radar
    )

    valid = True
    while plant.current_time() < self.duration:
      speed_lead = np.interp(plant.current_time(), self.speed_lead_breakpoints, self.speed_lead_values)
      log = plant.step(speed_lead)

      d_rel = log['distance_lead'] - log['distance'] if self.lead_relevancy else 200.
      v_rel = speed_lead - log['speed'] if self.lead_relevancy else 0.
      log['d_rel'] = d_rel
      log['v_rel'] = v_rel

      if d_rel < 1.0:
        print("Crashed!!!!")
        valid = False

    print("maneuver end", valid)
    return valid
