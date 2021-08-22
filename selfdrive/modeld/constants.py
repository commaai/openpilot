IDX_N = 33

def index_function(idx, max_val=192):
  return (max_val/1024)*(idx**2)


T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]
