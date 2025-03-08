from tinygrad import Device
from tinygrad.helpers import getenv, DEBUG, BEAM
from tinygrad.engine.search import beam_search, time_linearizer, bufs_from_lin
from extra.optimization.helpers import load_worlds, ast_str_to_lin

if __name__ == "__main__":
  filter_reduce = bool(getenv("FILTER_REDUCE"))
  ast_strs = load_worlds(filter_reduce=filter_reduce, filter_novariable=True)
  dev = Device[Device.DEFAULT]

  test_n = getenv("TEST_N", 10)
  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  beam_won, tested = 0, 0

  for num, ast in enumerate(ast_strs[:test_n]):
    def new_lin(): return ast_str_to_lin(ast, opts=dev.renderer)

    k = new_lin()
    # k.required_optimizations()

    if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()

    assert BEAM > 0

    lins = [(("tc" if used_tensor_cores else "hc"), k)]
    if used_tensor_cores:
      lins.append(("hc", new_lin()))
      lins[-1][1].hand_coded_optimizations()
    kb = new_lin()
    # kb.required_optimizations()
    test_rawbuffers = bufs_from_lin(kb)    # allocate scratch buffers for optimization
    lins.append((f"beam{BEAM.value}", beam_search(kb, test_rawbuffers, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))))
    timed = sorted([(nm, tk, time_linearizer(tk, test_rawbuffers, allow_test_size=False, clear_l2=True)) for nm, tk in lins], key=lambda x: x[2])
    if DEBUG >= 1: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))

    tested += 1
    if timed[0][0].startswith("beam"):
      beam_won += 1

  print(f"{beam_won=} / {tested=} = {beam_won/tested:.3f}")