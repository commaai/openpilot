from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin, time_linearizer, get_kernel_actions

if __name__ == "__main__":
  ast_strs = load_worlds()
  for i, ast_str in enumerate(ast_strs):
    lin = ast_str_to_lin(ast_str)
    rawbufs = bufs_from_lin(lin)
    test_tm = time_linearizer(lin, rawbufs)
    if test_tm < 1e-2: continue
    print(f"EXAMPLE {i}")
    acted_lins = get_kernel_actions(lin)
    ok_avg, short_avg = 0, 0
    for k,v in acted_lins.items():
      tm1 = time_linearizer(v, rawbufs)
      tm2 = time_linearizer(v, rawbufs)
      tm3 = time_linearizer(v, rawbufs, False)
      print(v.colored_shape(50), f"{tm1*1e3:10.2f} {tm2*1e3:10.2f} {tm3*1e3:10.2f}  :  {((tm1-tm2)/tm1)*100:5.2f}%  vs  {((tm1-tm3)/tm1)*100:5.2f}%")
      ok_avg += (tm1-tm2)/tm1
      short_avg += (tm1-tm3)/tm1
    print(f"{ok_avg/len(acted_lins)*100:5.2f}% vs {short_avg/len(acted_lins)*100:5.2f}%")
