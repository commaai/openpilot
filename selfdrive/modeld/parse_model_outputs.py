import numpy as np
from typing import Dict
from openpilot.selfdrive.modeld.constants import IDX_N

POSE_WIDTH = 6
SIM_POSE_WIDTH = 6
LEAD_WIDTH = 4
LANE_LINES_WIDTH = 2
ROAD_EDGES_WIDTH = 2
PLAN_WIDTH = 15
DESIRE_PRED_WIDTH = 4

NUM_LANE_LINES = 4
NUM_ROAD_EDGES = 2

LEAD_TRAJ_LEN = 6

PLAN_MHP_N = 5
LEAD_MHP_N = 2
PLAN_MHP_SELECTION = 1
LEAD_MHP_SELECTION = 3

FCW_THRESHOLD_5MS2_HIGH = 0.15
FCW_THRESHOLD_5MS2_LOW = 0.05
FCW_THRESHOLD_3MS2 = 0.7

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

def parse_mdn(name, outs, in_N=0, out_N=1, out_shape=None):
  if name not in outs:
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
  return

def parse_binary_crossentropy(name, outs):
  if name not in outs:
    return
  raw = outs[name]
  outs[name] = sigmoid(raw)
  return

def parse_categorical_crossentropy(name, outs, size=1):
  if name not in outs:
    return
  raw = outs[name]
  if size > 1:
    raw = raw.reshape((raw.shape[0], size, -1))
  outs[name] = softmax(raw, axis=-1)
  return

def parse_outputs(outs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
  parse_mdn('plan', outs, in_N=PLAN_MHP_N, out_N=PLAN_MHP_SELECTION, out_shape=(IDX_N,PLAN_WIDTH))
  parse_mdn('lane_lines', outs, in_N=0, out_N=0, out_shape=(NUM_LANE_LINES,IDX_N,LANE_LINES_WIDTH))
  parse_mdn('road_edges', outs, in_N=0, out_N=0, out_shape=(NUM_ROAD_EDGES,IDX_N,LANE_LINES_WIDTH))
  parse_mdn('pose', outs, in_N=0, out_N=0, out_shape=(POSE_WIDTH,))
  parse_mdn('road_transform', outs, in_N=0, out_N=0, out_shape=(POSE_WIDTH,))
  parse_mdn('sim_pose', outs, in_N=0, out_N=0, out_shape=(POSE_WIDTH,))
  parse_mdn('wide_from_device_euler', outs, in_N=0, out_N=0, out_shape=(POSE_WIDTH // 2,))
  parse_mdn('lead', outs, in_N=LEAD_MHP_N, out_N=LEAD_MHP_SELECTION, out_shape=(LEAD_TRAJ_LEN,LEAD_WIDTH))
  for k in ['lead_prob', 'lane_lines_prob', 'meta']:
    parse_binary_crossentropy(k, outs)
  parse_categorical_crossentropy('desire_pred', outs, size=DESIRE_PRED_WIDTH)
  return outs
