"""Free units in the plan head, who owns them, and a builder that writes circuits into them.

Plan head: x (512, x_plan_feat) -> r4 = relu(W_in x + b_in) (256) -> r6 = relu(W_r0 r4 + b_r0) (256)
  -> trunk = relu(r4 + W_r2 r6 + b_r2) (256) -> plan = (W_f trunk + b_f) * scale (990).

The units below never turned on in 129k replayed frames (111 segments, 11 routes, master model). patch_model.py
clears their wiring first (exact on that data), so a circuit in owned units can not leak into other units:
  W_r0[:, DEAD_R4] = 0   (dead layer-1 units feed no layer-2 unit)
  W_r2[:, DEAD_R6] = 0   (dead layer-2 units feed no trunk unit)
Each experiment owns its own units, so the patches stack. Circuit checks that a patch only writes what it owns.
"""
import json
import numpy as np

H = 'policy_model.temporal_hydra.'
N_PLAN = 990

DEAD_TRUNK = (11, 33, 47, 48, 51, 78, 108, 166, 227)
DEAD_R4 = (0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 23, 26, 28, 29, 32, 33, 37, 39, 40, 47, 48, 49, 50, 51, 58, 60, 61, 62, 64,
           68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 81, 85, 88, 92, 95, 99, 102, 108, 110, 114, 115, 116, 123, 124, 126, 128,
           133, 136, 140, 142, 148, 149, 152, 153, 154, 156, 158, 164, 166, 167, 169, 172, 173, 175, 178, 180, 181, 185,
           188, 194, 195, 198, 199, 201, 202, 203, 204, 207, 217, 219, 221, 222, 223, 224, 227, 228, 229, 230, 232, 237,
           239, 241, 244, 245, 246, 248, 252)
DEAD_R6 = (0, 2, 3, 5, 6, 10, 12, 16, 19, 24, 25, 26, 27, 38, 43, 45, 46, 47, 48, 53, 54, 55, 56, 59, 64, 68, 72, 74, 75,
           76, 77, 80, 81, 86, 97, 98, 100, 104, 105, 107, 109, 110, 114, 118, 123, 124, 126, 129, 139, 141, 143, 148, 149,
           150, 152, 157, 158, 165, 175, 178, 181, 185, 187, 191, 193, 199, 200, 201, 202, 204, 206, 207, 208, 211, 218,
           222, 225, 226, 231, 235, 236, 239, 240, 241, 249, 251, 252, 254)
SOFT_BRAKE_UNITS = (51, 78, 108, 166)  # trunk units used by the shipped soft brake

_R4_FREE = [j for j in DEAD_R4 if j not in DEAD_TRUNK]  # layer-1 slots whose trunk unit is live: need a skip cancel
OWNERS = ('lead', 'stop', 'cut', 'fork')
OWN = {name: {'r4': _R4_FREE[i::4], 'r6': list(DEAD_R6[i::4])} for i, name in enumerate(OWNERS)}
OWN['lead']['trunk'] = [227]
OWN['stop']['trunk'] = [47]
OWN['cut']['trunk'] = [11, 33]
OWN['fork']['trunk'] = [48]
LIVE_R4 = [j for j in range(256) if j not in DEAD_R4]


class Circuit:
  """Build a circuit in the units an experiment owns. Write with r4 / r6 / out, then save() for patch_model.py.

  r4(j, w_x, b):      layer-1 unit on x (512). If trunk j is live, an owned r6 unit cancels the skip path exactly
                      (it copies r4_j and subtracts it from trunk j), so trunk j does not change.
  r6(k, w, b):        layer-2 unit; w is a dict {r4 unit: weight}, live r4 units or own r4 slots only.
  out(j, w6, readout, w_x=None, b_x=0, b=0):
                      trunk unit j = relu(relu(w_x . x + b_x) + sum w6[k] r6_k + b), readout (990,) into the plan.
  """

  def __init__(self, owner):
    self.own = OWN[owner]
    self.owner = owner
    self.w = {}  # (tensor, flat index) -> value
    self.used = set()

  def _set(self, tensor, idx, val):
    for i, v in zip(np.atleast_1d(idx), np.atleast_1d(val), strict=True):
      self.w[(tensor, int(i))] = float(v)

  def _take(self, kind, unit):
    assert unit in self.own[kind], f'{self.owner} does not own {kind} unit {unit}'
    assert (kind, unit) not in self.used, f'{kind} unit {unit} already used'
    self.used.add((kind, unit))

  def free(self, kind):
    return [u for u in self.own[kind] if (kind, u) not in self.used]

  def r4(self, j, w_x, b):
    self._take('r4', j)
    self._set('in_layer.plan.weight', j * 512 + np.arange(512), w_x)
    self._set('in_layer.plan.bias', j, b)
    if j not in DEAD_TRUNK:
      k = self.free('r6')[0]
      self._take('r6', k)
      cols = np.array(sorted(set(LIVE_R4) | {j}))  # dead r4 columns are cleared by patch_model.py, not written here
      self._set('res_layer.plan.0.weight', k * 256 + cols, (cols == j).astype(float))
      self._set('res_layer.plan.0.bias', k, 0.0)
      self._set('res_layer.plan.2.weight', j * 256 + k, -1.0)
    return j

  def r6(self, k, w, b):
    self._take('r6', k)
    for c in w:
      assert c in LIVE_R4 or c in self.own['r4'], f'r6 unit {k} may not read r4 unit {c}'
    row = np.zeros(256)
    for c, v in w.items():
      row[c] = v
    cols = np.array(sorted(set(LIVE_R4) | set(w)))
    self._set('res_layer.plan.0.weight', k * 256 + cols, row[cols])
    self._set('res_layer.plan.0.bias', k, b)
    return k

  def out(self, j, w6, readout, w_x=None, b_x=0.0, b=0.0):
    self._take('trunk', j)
    for k in w6:
      assert k in self.own['r6'], f'trunk unit {j} may only read own r6 units, not {k}'
    self._set('in_layer.plan.weight', j * 512 + np.arange(512), np.zeros(512) if w_x is None else w_x)
    self._set('in_layer.plan.bias', j, b_x)
    row = np.zeros(256)
    for k, v in w6.items():
      row[k] = v
    self._set('res_layer.plan.2.weight', j * 256 + np.arange(256), row)
    self._set('res_layer.plan.2.bias', j, b)
    readout = np.asarray(readout, dtype=np.float64)
    assert readout.shape == (N_PLAN,)
    self._set('final_layer.plan.weight', np.arange(N_PLAN) * 256 + j, readout)
    return j

  def edits(self):
    by = {}
    for (t, i), v in sorted(self.w.items()):
      by.setdefault(t, ([], []))
      by[t][0].append(i)
      by[t][1].append(v)
    return [(H + t, np.array(i, dtype=np.int64), np.array(v, dtype=np.float32)) for t, (i, v) in by.items()]

  def save(self, out, desc):
    edits = self.edits()
    arrays = {}
    for i, (_, idx, val) in enumerate(edits):
      arrays[f'idx{i}'] = idx
      arrays[f'val{i}'] = val
    meta = {'name': self.owner, 'desc': desc, 'edits': [{'name': n} for n, _, _ in edits]}
    np.savez(out, meta=json.dumps(meta), **arrays)
    print(f'{out}: {sum(len(i) for _, i, _ in edits)} weights')
