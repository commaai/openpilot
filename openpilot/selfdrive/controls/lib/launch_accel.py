import numpy as np

# pulling away: the model asks for a gentle start, make it go when it decides to go.
# At speed, a small boost on the model's accel only (its slowdowns stay as they are).
LAUNCH_SPEED_BP = [0.0, 5.0]
LAUNCH_GAIN_V = [1.6, 1.1]
LAUNCH_MIN_ACCEL = 0.2


def get_launch_accel(a_e2e: float, v_ego: float) -> float:
  """The model's accel, with more of it when it speeds up, the most when the car pulls away. Braking is as is."""
  if a_e2e > LAUNCH_MIN_ACCEL:
    return float(np.interp(v_ego, LAUNCH_SPEED_BP, LAUNCH_GAIN_V)) * a_e2e
  return a_e2e
