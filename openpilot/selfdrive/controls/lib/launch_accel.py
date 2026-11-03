import numpy as np

# pulling away: the model asks for a gentle start, make it go when it decides to go
LAUNCH_SPEED_BP = [0.0, 5.0]
LAUNCH_GAIN_V = [1.6, 1.0]
LAUNCH_MIN_ACCEL = 0.2


def get_launch_accel(a_e2e: float, v_ego: float) -> float:
  """The model's accel, with more of it when the car pulls away. Everything else is the model's accel as is."""
  if a_e2e > LAUNCH_MIN_ACCEL:
    return float(np.interp(v_ego, LAUNCH_SPEED_BP, LAUNCH_GAIN_V)) * a_e2e
  return a_e2e
