import numpy as np

IDX_N = 33

def index_function(idx, max_val=192, max_idx=32):
  return (max_val) * ((idx/max_idx)**2)


T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]
X_IDXS = [index_function(idx, max_val=192.0) for idx in range(IDX_N)]
LEAD_T_IDXS = [0., 2., 4., 6., 8., 10.]
LEAD_T_OFFSETS = [0., 2., 4.]
META_T_IDXS = [2., 4., 6., 8., 10.]

FCW_THRESHOLDS_5MS2 = np.array([.05, .05, .15, .15, .15], dtype=np.float32)
FCW_THRESHOLDS_3MS2 = np.array([.7, .7], dtype=np.float32)

class Plan:
  POSITION = slice(0, 3)
  VELOCITY = slice(3, 6)
  ACCELERATION = slice(6, 9)
  T_FROM_CURRENT_EULER = slice(9, 12)
  ORIENTATION_RATE = slice(12, 15)

class Meta:
  ENGAGED = slice(0, 1)
  # next 2, 4, 6, 8, 10 seconds
  GAS_DISENGAGE = slice(1, 36, 7)
  BRAKE_DISENGAGE = slice(2, 36, 7)
  STEER_OVERRIDE = slice(3, 36, 7)
  HARD_BRAKE_3 = slice(4, 36, 7)
  HARD_BRAKE_4 = slice(5, 36, 7)
  HARD_BRAKE_5 = slice(6, 36, 7)
  GAS_PRESS = slice(7, 36, 7)
  # next 0, 2, 4, 6, 8, 10 seconds
  LEFT_BLINKER = slice(36, 48, 2)
  RIGHT_BLINKER = slice(37, 48, 2)