import numpy as np
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant


class Maneuver:
  def __init__(self, title, duration, **kwargs):
    # Was tempted to make a builder class
    self.distance_lead = kwargs.get("initial_distance_lead", 200.0)
    self.speed = kwargs.get("initial_speed", 0.0)
    self.lead_relevancy = kwargs.get("lead_relevancy", 0)

    self.breakpoints = kwargs.get("breakpoints", [0.0, duration])
    self.speed_lead_values = kwargs.get("speed_lead_values", [0.0 for i in range(len(self.breakpoints))])
    self.prob_lead_values = kwargs.get("prob_lead_values", [1.0 for i in range(len(self.breakpoints))])
    self.prob_throttle_values = kwargs.get("prob_throttle_values", [1.0 for i in range(len(self.breakpoints))])
    self.cruise_values = kwargs.get("cruise_values", [50.0 for i in range(len(self.breakpoints))])
    self.pitch_values = kwargs.get("pitch_values", [0.0 for i in range(len(self.breakpoints))])

    self.only_lead2 = kwargs.get("only_lead2", False)
    self.only_radar = kwargs.get("only_radar", False)
    self.ensure_start = kwargs.get("ensure_start", False)
    self.ensure_slowdown = kwargs.get("ensure_slowdown", False)
    self.enabled = kwargs.get("enabled", True)
    self.e2e = kwargs.get("e2e", False)
    self.personality = kwargs.get("personality", 0)
    self.force_decel = kwargs.get("force_decel", False)

    self.duration = duration
    self.title = title

  def evaluate(self):
    plant = Plant(
      lead_relevancy=self.lead_relevancy,
      speed=self.speed,
      distance_lead=self.distance_lead,
      enabled=self.enabled,
      only_lead2=self.only_lead2,
      only_radar=self.only_radar,
      e2e=self.e2e,
      personality=self.personality,
      force_decel=self.force_decel,
    )

    valid = True
    logs = []
    while plant.current_time < self.duration:
      speed_lead = np.interp(plant.current_time, self.breakpoints, self.speed_lead_values)
      prob_lead = np.interp(plant.current_time, self.breakpoints, self.prob_lead_values)
      cruise = np.interp(plant.current_time, self.breakpoints, self.cruise_values)
      pitch = np.interp(plant.current_time, self.breakpoints, self.pitch_values)
      prob_throttle = np.interp(plant.current_time, self.breakpoints, self.prob_throttle_values)
      log = plant.step(speed_lead, prob_lead, cruise, pitch, prob_throttle)

      d_rel = log['distance_lead'] - log['distance'] if self.lead_relevancy else 200.
      v_rel = speed_lead - log['speed'] if self.lead_relevancy else 0.
      log['d_rel'] = d_rel
      log['v_rel'] = v_rel
      logs.append(np.array([plant.current_time,
                            log['distance'],
                            log['distance_lead'],
                            log['speed'],
                            speed_lead,
                            log['acceleration']]))

      if d_rel < .4 and (self.only_radar or prob_lead > 0.5):
        print("Crashed!!!!")
        valid = False

      if self.ensure_start and log['v_rel'] > 0 and log['acceleration'] < 1e-3:
        print('LongitudinalPlanner not starting!')
        valid = False

    if self.ensure_slowdown and log['speed'] > 5.5:
      print('LongitudinalPlanner not slowing down!')
      valid = False

    if self.force_decel and log['speed'] > 1e-1 and log['acceleration'] > -0.04:
      print('Not stopping with force decel')
      valid = False


    print("maneuver end", valid)
    return valid, np.array(logs)
