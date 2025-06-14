import os
import numpy as np
import math, random
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.engine.search import actions, bufs_from_lin, get_kernel_actions
from tinygrad.nn.optim import Adam
from extra.optimization.extract_policynet import PolicyNet
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats, time_linearizer

if __name__ == "__main__":
  net = PolicyNet()
  if os.path.isfile("/tmp/policynet.safetensors"): load_state_dict(net, safe_load("/tmp/policynet.safetensors"))
  optim = Adam(get_parameters(net))

  ast_strs = load_worlds()

  # select a world
  all_feats, all_acts, all_rews = [], [], []
  while 1:
    Tensor.training = False
    lin = ast_str_to_lin(random.choice(ast_strs))
    rawbufs = bufs_from_lin(lin)
    tm = last_tm = base_tm = time_linearizer(lin, rawbufs)

    # take actions
    feats, acts, rews = [], [], []
    while 1:
      feat = lin_to_feats(lin)
      feats.append(feat)
      probs = net(Tensor([feat])).exp()[0].numpy()

      # mask valid actions
      valid_action_mask = np.zeros((len(actions)+1), dtype=np.float32)
      for x in get_kernel_actions(lin): valid_action_mask[x] = 1
      probs *= valid_action_mask
      probs /= sum(probs)

      act = np.random.choice(len(probs), p=probs)
      acts.append(act)
      if act == 0:
        rews.append(0)
        break
      try:
        lin.apply_opt(actions[act-1])
        tm = time_linearizer(lin, rawbufs)
        if math.isinf(tm): raise Exception("failed")
        rews.append(((last_tm-tm)/base_tm))
        last_tm = tm
      except Exception:
        rews.append(-0.5)
        break
      #print(f"{tm*1e6:10.2f}", lin.colored_shape())

    assert len(feats) == len(acts) and len(acts) == len(rews)
    #print(rews)
    print(f"***** EPISODE {len(rews)} steps, {sum(rews):5.2f} reward, {base_tm*1e6:12.2f} -> {tm*1e6:12.2f} : {lin.colored_shape()}")
    all_feats += feats
    all_acts += acts
    # rewards to go
    for i in range(len(rews)-2, -1, -1): rews[i] += rews[i+1]
    all_rews += rews

    BS = 32
    if len(all_feats) >= BS:
      Tensor.training = True
      x = Tensor(all_feats[:BS])
      mask = np.zeros((BS, len(actions)+1), dtype=np.float32)
      mask[range(BS), all_acts[:BS]] = all_rews[:BS]
      loss = -(net(x) * Tensor(mask)).mean()
      optim.zero_grad()
      loss.backward()
      optim.step()
      all_feats = all_feats[BS:]
      all_acts = all_acts[BS:]
      all_rews = all_rews[BS:]
