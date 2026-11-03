import numpy as np

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter

Personality = log.LongitudinalPersonality

ACCEL_GAIN = 1.2
LIFT_PER_SPEED = 0.012  # m/s^2 of lift per m/s of speed
LIFT_RC = 0.3
PROB_IDX = 4  # gas and brake press probabilities 8s ahead
GATE_BP = [0.6, 0.9]  # fleet data: below 0.6, brake presses follow as often as gas presses
# pulling away: the model asks for a gentle start, make it go when it decides to go
LAUNCH_SPEED_BP = [0.0, 5.0]
LAUNCH_GAIN_V = [1.6, 1.0]
LAUNCH_MIN_ACCEL = 0.2


class PersonalityAccel:
  """Scales the model's accel by personality: relaxed is unchanged, standard adds gain, aggressive adds gain and lift."""

  def __init__(self, dt):
    self.lift_filter = FirstOrderFilter(0.0, LIFT_RC, dt)

  def reset(self):
    self.lift_filter.x = 0.0

  def update(self, sm, a_e2e):
    personality = sm['selfdriveState'].personality
    predictions = sm['modelV2'].meta.disengagePredictions
    p_gas, p_brake = predictions.gasPressProbs, predictions.brakePressProbs

    lift_target = 0.0
    if personality == Personality.aggressive and a_e2e >= 0.0 and len(p_gas) > PROB_IDX and len(p_brake) > PROB_IDX:
      # The model predicting a gas press and no brake press means it isn't slowing for anything
      gate = np.interp(p_gas[PROB_IDX] * (1.0 - p_brake[PROB_IDX]), GATE_BP, [0.0, 1.0])
      lift_target = LIFT_PER_SPEED * max(sm['carState'].vEgo, 0.0) * gate
    lift = self.lift_filter.update(lift_target)

    gain = 1.0 if personality == Personality.relaxed else ACCEL_GAIN
    if a_e2e > LAUNCH_MIN_ACCEL:
      gain *= np.interp(sm['carState'].vEgo, LAUNCH_SPEED_BP, LAUNCH_GAIN_V)
    return gain * a_e2e + lift
