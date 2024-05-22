import numpy as np
from openpilot.selfdrive.modeld.constants import ModelConstants

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1):
  x -= np.max(x, axis=axis, keepdims=True)
  if x.dtype == np.float32 or x.dtype == np.float64:
    np.exp(x, out=x)
  else:
    x = np.exp(x)
  x /= np.sum(x, axis=axis, keepdims=True)
  return x

class Parser:
  def __init__(self, ignore_missing=False):
    self.ignore_missing = ignore_missing

  def check_missing(self, outs, name):
    if name not in outs and not self.ignore_missing:
      raise ValueError(f"Missing output {name}")
    return name not in outs

  def parse_categorical_crossentropy(self, name, outs, out_shape=None):
    if self.check_missing(outs, name):
      return
    raw = outs[name]
    if out_shape is not None:
      raw = raw.reshape((raw.shape[0],) + out_shape)
    outs[name] = softmax(raw, axis=-1)

  def parse_binary_crossentropy(self, name, outs):
    if self.check_missing(outs, name):
      return
    raw = outs[name]
    outs[name] = sigmoid(raw)

  def parse_mdn(self, name, outs, in_N=0, out_N=1, out_shape=None):
    if self.check_missing(outs, name):
      return
    raw = outs[name]
    raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))

    pred_mu = raw[:,:,:(raw.shape[2] - out_N)//2]
    n_values = (raw.shape[2] - out_N)//2
    pred_mu = raw[:,:,:n_values]
    pred_std = np.exp(raw[:,:,n_values: 2*n_values])

    if in_N > 1:
      weights = np.zeros((raw.shape[0], in_N, out_N), dtype=raw.dtype)
      for i in range(out_N):
        weights[:,:,i - out_N] = softmax(raw[:,:,i - out_N], axis=-1)

      if out_N == 1:
        for fidx in range(weights.shape[0]):
          idxs = np.argsort(weights[fidx][:,0])[::-1]
          weights[fidx] = weights[fidx][idxs]
          pred_mu[fidx] = pred_mu[fidx][idxs]
          pred_std[fidx] = pred_std[fidx][idxs]
      full_shape = tuple([raw.shape[0], in_N] + list(out_shape))
      outs[name + '_weights'] = weights
      outs[name + '_hypotheses'] = pred_mu.reshape(full_shape)
      outs[name + '_stds_hypotheses'] = pred_std.reshape(full_shape)

      pred_mu_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
      pred_std_final = np.zeros((raw.shape[0], out_N, n_values), dtype=raw.dtype)
      for fidx in range(weights.shape[0]):
        for hidx in range(out_N):
          idxs = np.argsort(weights[fidx,:,hidx])[::-1]
          pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
          pred_std_final[fidx, hidx] = pred_std[fidx, idxs[0]]
    else:
      pred_mu_final = pred_mu
      pred_std_final = pred_std

    if out_N > 1:
      final_shape = tuple([raw.shape[0], out_N] + list(out_shape))
    else:
      final_shape = tuple([raw.shape[0],] + list(out_shape))
    outs[name] = pred_mu_final.reshape(final_shape)
    outs[name + '_stds'] = pred_std_final.reshape(final_shape)

  def parse_outputs(self, outs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    self.parse_mdn('plan', outs, in_N=ModelConstants.PLAN_MHP_N, out_N=ModelConstants.PLAN_MHP_SELECTION,
                   out_shape=(ModelConstants.IDX_N,ModelConstants.PLAN_WIDTH))
    self.parse_mdn('lane_lines', outs, in_N=0, out_N=0, out_shape=(ModelConstants.NUM_LANE_LINES,ModelConstants.IDX_N,ModelConstants.LANE_LINES_WIDTH))
    self.parse_mdn('road_edges', outs, in_N=0, out_N=0, out_shape=(ModelConstants.NUM_ROAD_EDGES,ModelConstants.IDX_N,ModelConstants.LANE_LINES_WIDTH))
    self.parse_mdn('pose', outs, in_N=0, out_N=0, out_shape=(ModelConstants.POSE_WIDTH,))
    self.parse_mdn('road_transform', outs, in_N=0, out_N=0, out_shape=(ModelConstants.POSE_WIDTH,))
    self.parse_mdn('sim_pose', outs, in_N=0, out_N=0, out_shape=(ModelConstants.POSE_WIDTH,))
    self.parse_mdn('wide_from_device_euler', outs, in_N=0, out_N=0, out_shape=(ModelConstants.WIDE_FROM_DEVICE_WIDTH,))
    self.parse_mdn('lead', outs, in_N=ModelConstants.LEAD_MHP_N, out_N=ModelConstants.LEAD_MHP_SELECTION,
                   out_shape=(ModelConstants.LEAD_TRAJ_LEN,ModelConstants.LEAD_WIDTH))
    if 'lat_planner_solution' in outs:
      self.parse_mdn('lat_planner_solution', outs, in_N=0, out_N=0, out_shape=(ModelConstants.IDX_N,ModelConstants.LAT_PLANNER_SOLUTION_WIDTH))
    if 'desired_curvature' in outs:
      self.parse_mdn('desired_curvature', outs, in_N=0, out_N=0, out_shape=(ModelConstants.DESIRED_CURV_WIDTH,))
    for k in ['lead_prob', 'lane_lines_prob', 'meta']:
      self.parse_binary_crossentropy(k, outs)
    self.parse_categorical_crossentropy('desire_state', outs, out_shape=(ModelConstants.DESIRE_PRED_WIDTH,))
    self.parse_categorical_crossentropy('desire_pred', outs, out_shape=(ModelConstants.DESIRE_PRED_LEN,ModelConstants.DESIRE_PRED_WIDTH))
    return outs
