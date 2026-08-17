from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType

BLOCK_ROW = 256

def _sharded_invalids(shape:tuple[int, ...], dtype, device) -> Tensor:
  if isinstance(device, tuple):
    per = Tensor.invalids(shape[0]//len(device), *shape[1:], dtype=dtype, device=device)
    return Tensor(per.uop.unshard(0), device=device)
  return Tensor.invalids(*shape, dtype=dtype, device=device)

def _atomic_add(device:str) -> str:
  return "__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);" if device == "AMD" \
    else "__atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED);"

def _blk_for(D:int) -> int:
  blk = 64
  while D % blk: blk //= 2
  return blk

def _kv_ranges(G, N, D, BLK):
  g = UOp.range(G, 0)
  m = UOp.range(N, 1)
  jo = UOp.range(D // BLK, 2)
  ji = UOp.range(BLK, 3, AxisType.LOCAL)
  return g, m, jo * BLK + ji, jo, ji

def _ggather_fwd_kernel(out:UOp, table:UOp, idx:UOp) -> UOp:
  G, M, D = out.shape
  g, m, j, jo, ji = _kv_ranges(G, M, D, _blk_for(D))
  row = idx.index(g, m).cast(dtypes.weakint)
  val = table.index(g, row, j).load()
  return out.index(g, m, j).store(val).end(g, m, jo, ji).sink(
    arg=KernelInfo(name=f"ggather_fwd_{M}_{D}", opts_to_apply=()))

def _ggather_zero_kernel(out:UOp) -> UOp:
  i = UOp.range(out.numel(), 0)
  return out.flatten().index(i).store(UOp.const(0.0, out.dtype)).end(i).sink(arg=KernelInfo(name="ggather_zero"))

def _sharded_zeros(shape:tuple[int, ...], dtype, device) -> Tensor:
  return Tensor.custom_kernel(_sharded_invalids(shape, dtype, device), fxn=_ggather_zero_kernel)[0]

def _ggather_bwd(gradient:UOp, kernel:UOp) -> tuple:
  _, table_u, idx_u = kernel.src[1:4]
  dev = table_u.device
  device = (dev[0] if isinstance(dev, tuple) else dev).split(":")[0]
  G, R, D = table_u.shape
  gt = _sharded_zeros((G, R, D), dtypes.float32, dev)
  go = Tensor(gradient, device=dev)
  atomic_str = _atomic_add(device)
  def _bwd_kernel(gtab:UOp, gout:UOp, idx:UOp) -> UOp:
    Gk, M, Dk = gout.shape
    g, m, j, jo, ji = _kv_ranges(Gk, M, Dk, _blk_for(Dk))
    row = idx.index(g, m).cast(dtypes.weakint)
    val = gout.index(g, m, j).load().cast(dtypes.float32)
    atomic = UOp(Ops.CUSTOM, dtypes.void, (gtab.index(g, row, j), val), arg=atomic_str)
    return atomic.end(g, m, jo, ji).sink(arg=KernelInfo(name=f"ggather_bwd_{M}_{Dk}", opts_to_apply=()))
  grad_table = Tensor.custom_kernel(gt, go, Tensor(idx_u, device=dev), fxn=_bwd_kernel)[0]
  return (None, grad_table.cast(table_u.dtype).uop, None)

def grouped_gather_rows(table:Tensor, idx:Tensor, n_groups:int) -> Tensor:
  G, R, D = table.shape
  M = idx.shape[1]
  out = _sharded_invalids((G, M, D), table.dtype, table.device)
  return Tensor.custom_kernel(out, table, idx, fxn=_ggather_fwd_kernel, grad_fxn=_ggather_bwd)[0]

def _gscatter_fwd_kernel(out:UOp, src:UOp, idx:UOp) -> UOp:
  G, M, D = out.shape
  k = idx.shape[1] // src.shape[1]
  g, m, j, jo, ji = _kv_ranges(G, idx.shape[1], D, _blk_for(D))
  row = idx.index(g, m).cast(dtypes.weakint)
  val = src.index(g, (m // k).cast(dtypes.weakint), j).load()
  return out.index(g, row, j).store(val).end(g, m, jo, ji).sink(
    arg=KernelInfo(name=f"gscatter_fwd_{idx.shape[1]}_{D}", opts_to_apply=()))

def _gscatter_bwd(gradient:UOp, kernel:UOp) -> tuple:
  _, src_u, idx_u = kernel.src[1:4]
  dev = src_u.device
  G, T_l, D = src_u.shape
  k = idx_u.shape[1] // T_l
  sel = grouped_gather_rows(Tensor(gradient, device=dev), Tensor(idx_u, device=dev), G)
  return (None, sel.reshape(G, T_l, k, D).sum(2).cast(src_u.dtype).uop, None)

def grouped_scatter_rows(src:Tensor, idx:Tensor, m_l:int) -> Tensor:
  G, T_l, D = src.shape
  zero = _sharded_zeros((G, m_l, D), src.dtype, src.device)
  return Tensor.custom_kernel(zero, src, idx, fxn=_gscatter_fwd_kernel, grad_fxn=_gscatter_bwd)[0]

def m_max_for(t_local:int, experts_per_tok:int, n_experts:int) -> int:
  return (-(-t_local * experts_per_tok // BLOCK_ROW) + n_experts) * BLOCK_ROW

class Routing:
  def __init__(self, weights:Tensor, dest_row:Tensor, off:Tensor, m_l:int, n_groups:int, t_local:int):
    self.weights, self.dest_row = weights, dest_row
    self.off = off
    self.m_l, self.n_groups, self.t_local = m_l, n_groups, t_local

  @property
  def rows_e(self) -> Tensor:
    G, E = self.off.shape[0], self.off.shape[1] - 1
    tr = Tensor.arange(self.m_l // BLOCK_ROW, dtype=dtypes.int32).reshape(1, -1, 1) * BLOCK_ROW
    tr = tr.shard(self.off.device) if isinstance(self.off.device, tuple) else tr.to(self.off.device)
    tile_e = ((tr >= self.off[:, :E].reshape(G, 1, E)).sum(-1) - 1).cast(dtypes.int32)
    return tile_e.reshape(-1, 1).expand(-1, BLOCK_ROW).reshape(-1)

def n_groups_of(t:Tensor) -> int:
  return len(t.device) if isinstance(t.device, tuple) else 1

def route(logits:Tensor, experts_per_tok:int, n_experts:int) -> Routing:
  T, E = logits.shape
  k, G = experts_per_tok, n_groups_of(logits)
  assert T % G == 0, f"tokens {T} must split across {G} devices"
  T_l, m_l = T // G, m_max_for(T // G, k, n_experts)

  topv, topi = logits.reshape(G, T_l, E).topk(k)
  weights = topv.softmax(-1)
  m = topi.reshape(G, T_l * k).cast(dtypes.int32).one_hot(E).cast(dtypes.int32)

  pad = ((m.sum(1) + (BLOCK_ROW - 1)) // BLOCK_ROW) * BLOCK_ROW
  off = pad.cumsum(1).pad(((0, 0), (1, 0)))
  dest_row = ((m.cumsum(1) + off[:, :E].reshape(G, 1, E)) * m).sum(-1).sub(1).cast(dtypes.int32)
  return Routing(weights, dest_row, off, m_l, G, T_l)

def dispatch(x:Tensor, r:Routing) -> Tensor:
  G, D = r.n_groups, x.shape[-1]
  return grouped_scatter_rows(x.reshape(G, r.t_local, D), r.dest_row, r.m_l).reshape(G * r.m_l, D)

def combine(y:Tensor, r:Routing, n_tokens:int, experts_per_tok:int) -> Tensor:
  G, D, k = r.n_groups, y.shape[-1], experts_per_tok
  sel = grouped_gather_rows(y.reshape(G, r.m_l, D), r.dest_row, G).reshape(G, r.t_local, k, D)
  return (sel * r.weights.reshape(G, r.t_local, k, 1).cast(sel.dtype)).sum(2).reshape(n_tokens, D).cast(y.dtype)
