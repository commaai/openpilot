import random
from tinygrad.helpers import getenv
from tinygrad.engine.search import beam_search, bufs_from_lin
from tinygrad.codegen.heuristic import hand_coded_optimizations
from extra.optimization.helpers import load_worlds, ast_str_to_lin, time_linearizer

def optimize_kernel(k):
  # TODO: update this
  return hand_coded_optimizations(k)

if __name__ == '__main__':
  hcopt_wins = beam_wins = tie = 0
  hcopt_total = beam_total = 0.0

  worlds = load_worlds(filter_reduce=False, filter_noimage=True, filter_novariable=False)
  random.seed(0)
  random.shuffle(worlds)

  for world in worlds[:500]:
    k = ast_str_to_lin(world)
    rawbufs = bufs_from_lin(k)

    k_hcopt = k.copy()
    k_hcopt.apply_opts(optimize_kernel(k_hcopt))
    k_beam = beam_search(k.copy(), rawbufs, getenv("BEAM", 2))

    disable_cache = bool(getenv("NOCACHE", 0))
    t_hcopt = time_linearizer(k_hcopt, rawbufs, allow_test_size=False, cnt=10, disable_cache=disable_cache, clear_l2=True) * 1e6
    t_beam = time_linearizer(k_beam, rawbufs, allow_test_size=False, cnt=10, disable_cache=disable_cache, clear_l2=True) * 1e6

    if t_hcopt == t_beam: tie += 1
    elif t_hcopt < t_beam: hcopt_wins += 1
    else: beam_wins += 1
    hcopt_total += t_hcopt
    beam_total += t_beam

    print(f"{t_hcopt=:5.2f} {k_hcopt.applied_opts=}")
    print("")
    print(f"{t_beam=:5.2f} {k_beam.applied_opts=}")
    print("*"*20)

  print(f"{hcopt_wins=}, {beam_wins=}, {tie=}")
  print(f"{hcopt_total=:.2f}, {beam_total=:.2f}")