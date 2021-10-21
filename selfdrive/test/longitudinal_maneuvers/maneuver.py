import numpy as np
from selfdrive.test.longitudinal_maneuvers.plant import Plant


class Maneuver():
  def __init__(self, title, duration, **kwargs):
    # Was tempted to make a builder class
    self.distance_lead = kwargs.get("initial_distance_lead", 200.0)
    self.speed = kwargs.get("initial_speed", 0.0)
    self.lead_relevancy = kwargs.get("lead_relevancy", 0)

    self.breakpoints = kwargs.get("breakpoints", [0.0, duration])
    self.speed_lead_values = kwargs.get("speed_lead_values", [0.0 for i in range(len(self.breakpoints))])
    self.prob_lead_values = kwargs.get("prob_lead_values", [1.0 for i in range(len(self.breakpoints))])
    self.cruise_values = kwargs.get("cruise_values", [50.0 for i in range(len(self.breakpoints))])

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
    logs = []
    while plant.current_time() < self.duration:
      speed_lead = np.interp(plant.current_time(), self.breakpoints, self.speed_lead_values)
      prob = np.interp(plant.current_time(), self.breakpoints, self.prob_lead_values)
      cruise = np.interp(plant.current_time(), self.breakpoints, self.cruise_values)
      log = plant.step(speed_lead, prob, cruise)

      d_rel = log['distance_lead'] - log['distance'] if self.lead_relevancy else 200.
      v_rel = speed_lead - log['speed'] if self.lead_relevancy else 0.
      log['d_rel'] = d_rel
      log['v_rel'] = v_rel
      logs.append(np.array([plant.current_time(),
                            log['distance'],
                            log['distance_lead'],
                            log['speed'],
                            speed_lead,
                            log['acceleration']]))

      if d_rel < .4 and (self.only_radar or prob > 0.5):
        print("Crashed!!!!")
        valid = False

    print("maneuver end", valid)
    return valid, np.array(logs)
