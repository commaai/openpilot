import numpy as np
from openpilot.common.realtime import DT_CTRL, DT_MDL

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# PID output bound for LatControlCurvature (not used by torque-controlled cars).
MAX_CURVATURE = 0.2
MIN_STABLE_DELAY = 0.3

# Experimental limit: 3x the original 5 m/s^3.
MAX_LATERAL_JERK = 15.0  # m/s^3

def should_stop(v_ego: float, a_target: float) -> bool:
  return bool(v_ego < 0.3 and a_target < 0.1)

def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val

def clip_curvature(v_ego, prev_curvature, new_curvature) -> tuple[float, bool]:
  # Limit lateral jerk only; no fixed curvature or lateral-acceleration cap.
  v_ego = max(v_ego, MIN_SPEED)
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)  # Approximation at constant speed.
  return clamp(new_curvature,
               prev_curvature - max_curvature_rate * DT_CTRL,
               prev_curvature + max_curvature_rate * DT_CTRL)


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    if action_t < MIN_STABLE_DELAY:
      v_target = v_now + (action_t / MIN_STABLE_DELAY) * (np.interp(MIN_STABLE_DELAY, t_idxs, speeds) - v_now)
    else:
      v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
  else:
    a_target = 0.0
  return a_target

def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  if action_t < MIN_STABLE_DELAY:
    psi_target = (action_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, t_idxs, yaws)
  else:
    psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)
