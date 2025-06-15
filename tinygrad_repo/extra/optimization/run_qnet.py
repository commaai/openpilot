from typing import List, Tuple
from tinygrad.codegen.kernel import Kernel
from tinygrad.engine.search import get_kernel_actions, actions

_net = None
def beam_q_estimate(beam:List[Tuple[Kernel, float]]) -> List[Tuple[Kernel, float]]:
  global _net
  if _net is None:
    from tinygrad.nn.state import load_state_dict, safe_load
    from extra.optimization.pretrain_valuenet import ValueNet
    _net = ValueNet(1021+len(actions), 2)
    load_state_dict(_net, safe_load("/tmp/qnet.safetensors"), verbose=False)
  from tinygrad.tensor import Tensor
  from tinygrad.helpers import Context
  from extra.optimization.helpers import lin_to_feats
  import numpy as np
  feats = []
  lins = []
  base_tms = []
  for lin,tm in beam:
    lin_feats = lin_to_feats(lin)
    for a,v in get_kernel_actions(lin, include_0=False).items():
      acts = np.zeros(len(actions))
      acts[a-1] = 1.0
      feats.append(np.concatenate([lin_feats, acts]))
      lins.append(v)
      base_tms.append(tm)
  with Context(BEAM=0):
    with Tensor.train(False):
      preds = _net(Tensor(feats)).numpy()
  pred_time = np.array(base_tms) / np.exp(preds[:, 0])
  return sorted(zip(lins, pred_time), key=lambda x: x[1])
