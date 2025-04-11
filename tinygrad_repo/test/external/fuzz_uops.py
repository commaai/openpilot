import itertools
from collections import defaultdict
import numpy as np
from dataclasses import replace
from typing import DefaultDict, Dict, List, Tuple
from tinygrad.ops import UOp, print_uops, Ops
from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import DEBUG, colored
from tinygrad.ops import Variable
from tinygrad.tensor import _to_np_dtype
from test.external.fuzz_schedule import FUZZ_SCHEDULE_MAX_PATHS, find_all_toposorts

END_FOR_UOP = {Ops.IF:(Ops.STORE, Ops.ENDIF), Ops.RANGE:(Ops.ASSIGN, Ops.ENDRANGE)}

def fuzz_uops(uops:List[UOp]) -> List[Tuple[UOp, ...]]:
  blocks: List[List[UOp]] = [[]]
  for u in uops:
    if u.op in END_FOR_UOP: blocks.append([u])
    elif u.op in {x[1] for x in END_FOR_UOP.values()}: blocks.extend([[u], []])
    else: blocks[-1].append(u)

  paths_for_block: Dict[int, List[Tuple[UOp, ...]]] = {}
  for bi, bb in enumerate(blocks):
    children: DefaultDict[UOp, List[UOp]] = defaultdict(list)
    in_degree: Dict[UOp, int] = {}
    for u in bb:
      in_degree[u] = 0
      for x in u.src:
        if x in bb:
          children[x].append(u)
          in_degree[u] += 1
    paths_for_block[bi] = find_all_toposorts(children, in_degree)
  paths: Dict[Tuple[UOp, ...], None] = {}
  for up in itertools.product(*paths_for_block.values()):
    paths[tuple(uop for path in up for uop in path)] = None
    if len(paths) >= FUZZ_SCHEDULE_MAX_PATHS: break
  return list(paths)

class UOpsFuzzerRunner(CompiledRunner):
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False):
    assert self.p.uops is not None
    fuzz_paths = fuzz_uops(self.p.uops)
    init_rawbufs, init_name = {x:x.as_buffer() for x in rawbufs}, self.p.function_name
    init_globals = dict(zip(self.p.globals, rawbufs))
    if DEBUG >= 1: print(colored(f"fuzzing {len(fuzz_paths)} uop permutations for {init_name}", "yellow"))

    super().__call__(rawbufs, var_vals, wait)
    ground_truth = {x:np.frombuffer(x.as_buffer(), _to_np_dtype(x.dtype)) for x in rawbufs}

    for i, path in enumerate(fuzz_paths):
      # setup prg
      uops = list(path)
      if DEBUG >= 5: print_uops(uops)
      self.p = replace(self.p, name=(name:=f"{init_name}fuzz{i}"), src=Device[self.p.device].renderer.render(uops), uops=uops)
      if DEBUG >= 4: print(self.p.src)
      self.lib = Device[self.p.device].compiler.compile_cached(self.p.src)
      self.clprg = Device[self.p.device].runtime(name, self.lib)
      for x in (rawbufs:=[init_globals[i] for i in self.p.globals]): x.copyin(init_rawbufs[x])
      # verify
      super().__call__(rawbufs, var_vals, wait)
      for i, x in enumerate(rawbufs):
        try:
          np.testing.assert_allclose(np.frombuffer(x.as_buffer(), _to_np_dtype(x.dtype)), ground_truth[x], atol=1e-6, rtol=1e-6)
          if DEBUG >= 2: print(colored(name, "green"))
        except AssertionError as e:
          print(colored(name, "red"))
          raise e
