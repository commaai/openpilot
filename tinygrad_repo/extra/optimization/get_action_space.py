import random
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.opt.search import actions
from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.helpers import tqdm

tactions = set()
def test_rebuild(lin):
  linr = Kernel(lin.ast)
  for o in lin.applied_opts:
    assert o in actions, f"{o} is not in actions"
    tactions.add(o)
    linr.apply_opt(o)

  assert len(lin.sts) == len(linr.sts)
  for st1,st2 in zip(lin.sts, linr.sts):
    assert st1 == st2, f"{st1} != {st2}"

if __name__ == "__main__":
  ast_strs = load_worlds(False, False, False)
  random.shuffle(ast_strs)
  ast_strs = ast_strs[:2000]
  for ast_str in tqdm(ast_strs):
    lin = ast_str_to_lin(ast_str)
    #if not lin.apply_tensor_cores():
    lin.apply_opts(hand_coded_optimizations(lin))
    test_rebuild(lin)

  print(len(tactions), len(actions))
  print(sorted(list(tactions)))
