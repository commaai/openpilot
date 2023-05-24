IDX_N = 33

def index_function(idx, max_val=192, max_idx=32):
  return (max_val) * ((idx/max_idx)**2)


T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]
