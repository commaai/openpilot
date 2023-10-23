import numpy as np

def index_function(idx, max_val=192, max_idx=32):
  return (max_val) * ((idx/max_idx)**2)

class ModelConstants:
  # time and distance indices
  IDX_N = 33
  T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]
  X_IDXS = [index_function(idx, max_val=192.0) for idx in range(IDX_N)]
  LEAD_T_IDXS = [0., 2., 4., 6., 8., 10.]
  LEAD_T_OFFSETS = [0., 2., 4.]
  META_T_IDXS = [2., 4., 6., 8., 10.]

  # model inputs constants
  MODEL_FREQ = 20
  FEATURE_LEN = 512
  HISTORY_BUFFER_LEN = 99
  DESIRE_LEN = 8
  TRAFFIC_CONVENTION_LEN = 2
  NAV_FEATURE_LEN = 256
  NAV_INSTRUCTION_LEN = 150
  DRIVING_STYLE_LEN = 12

  # model outputs constants
  FCW_THRESHOLDS_5MS2 = np.array([.05, .05, .15, .15, .15], dtype=np.float32)
  FCW_THRESHOLDS_3MS2 = np.array([.7, .7], dtype=np.float32)
  FCW_5MS2_PROBS_WIDTH = 5
  FCW_3MS2_PROBS_WIDTH = 2

  DISENGAGE_WIDTH = 5
  POSE_WIDTH = 6
  WIDE_FROM_DEVICE_WIDTH = 3
  SIM_POSE_WIDTH = 6
  LEAD_WIDTH = 4
  LANE_LINES_WIDTH = 2
  ROAD_EDGES_WIDTH = 2
  PLAN_WIDTH = 15
  DESIRE_PRED_WIDTH = 8

  NUM_LANE_LINES = 4
  NUM_ROAD_EDGES = 2

  LEAD_TRAJ_LEN = 6
  DESIRE_PRED_LEN = 4

  PLAN_MHP_N = 5
  LEAD_MHP_N = 2
  PLAN_MHP_SELECTION = 1
  LEAD_MHP_SELECTION = 3

  FCW_THRESHOLD_5MS2_HIGH = 0.15
  FCW_THRESHOLD_5MS2_LOW = 0.05
  FCW_THRESHOLD_3MS2 = 0.7

  CONFIDENCE_BUFFER_LEN = 5
  RYG_GREEN = 0.01165
  RYG_YELLOW = 0.06157

# model outputs slices
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
