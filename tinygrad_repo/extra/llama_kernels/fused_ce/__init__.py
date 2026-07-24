import functools
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType

@functools.cache
def _custom_fused_ce_loss_fwd(loss_out:UOp, max_out:UOp, lse_out:UOp, logits:UOp, targets:UOp,
                              vocab:int, rows:int, seq:int, label_smoothing:float) -> UOp:
  row = UOp.range(rows, 0)
  b = row // seq
  s = row % seq

  v_max = UOp.range(vocab, 1, axis_type=AxisType.REDUCE)
  row_max = logits[b, s, v_max].cast(dtypes.float).reduce(v_max, arg=Ops.MAX)

  v_lse = UOp.range(vocab, 2, axis_type=AxisType.REDUCE)
  row_lse = (logits[b, s, v_lse].cast(dtypes.float) - row_max).exp().reduce(v_lse, arg=Ops.ADD).log() + row_max

  v_smooth = UOp.range(vocab, 3, axis_type=AxisType.REDUCE)
  target = logits[b, s, targets[row].cast(dtypes.weakint)].cast(dtypes.float)
  mean_logits = logits[b, s, v_smooth].cast(dtypes.float).reduce(v_smooth, arg=Ops.ADD) / vocab
  loss = row_lse - (1.0 - label_smoothing) * target - label_smoothing * mean_logits
  stores = UOp.group(loss_out[row].store(loss), max_out[row].store(row_max), lse_out[row].store(row_lse))

  return stores.end(row).sink(arg=KernelInfo(f"fused_ce_loss_fwd_{rows}_{vocab}"))

@functools.cache
def _custom_fused_ce_loss_bwd(d_logits:UOp, logits:UOp, lse:UOp, targets:UOp, scale:UOp,
                              vocab:int, rows:int, seq:int, label_smoothing:float) -> UOp:
  row = UOp.range(rows, 0)
  v = UOp.range(vocab, 1)
  b = row // seq
  s = row % seq

  prob = (logits[b, s, v].cast(dtypes.float) - lse[row]).exp()
  target = v.eq(targets[row].cast(dtypes.weakint)).where(1.0 - label_smoothing, 0.0)
  smooth = label_smoothing / vocab
  grad = (prob - target - smooth) * scale[0]

  return d_logits[b, s, v].store(grad.cast(d_logits.dtype)).end(v, row).sink(arg=KernelInfo(f"fused_ce_loss_bwd_{rows}_{vocab}"))

def _fused_ce_loss_bwd(gradient:UOp, kernel:UOp, label_smoothing:float):
  # NOTE: forward inputs are (loss_out, max_out, lse_out, logits, targets)
  # gradient is the upstream grad w.r.t. per-row loss (shape: (rows,) fp32)
  _, _, lse_u, logits_u, targets_u = kernel.src[1:]
  device = logits_u.device
  MBS, SEQ, VOCAB = logits_u.shape
  if isinstance(device, tuple):
    axis = logits_u.axis
    ndev = len(device)
    local_shape = tuple(s//ndev if i == axis else s for i,s in enumerate((MBS, SEQ, VOCAB)))
    d_logits = Tensor(Tensor.invalids(*local_shape, dtype=dtypes.bfloat16, device=device).uop.multi(axis), device=device)
    rows_per_dev = local_shape[0] * local_shape[1]
    seq_per_dev = local_shape[1]
  else:
    d_logits = Tensor.invalids(MBS, SEQ, VOCAB, dtype=dtypes.bfloat16, device=device)
    rows_per_dev = MBS * SEQ
    seq_per_dev = SEQ
  # NOTE: .mean() backward gives same grad per row (1/N), so broadcast is safe; take scalar
  scale = Tensor(gradient, device=device).float().reshape(-1)[0:1].contiguous()
  logits_t = Tensor(logits_u.after(kernel), device=device)
  lse_t = Tensor(lse_u.after(kernel), device=device)
  targets_t = Tensor(targets_u, device=device)
  fxn = functools.partial(_custom_fused_ce_loss_bwd, vocab=VOCAB, rows=rows_per_dev, seq=seq_per_dev, label_smoothing=label_smoothing)
  d_logits, *_ = Tensor.custom_kernel(d_logits, logits_t, lse_t, targets_t, scale, fxn=fxn)
  return (None, None, None, d_logits.uop, None)

def fused_ce_loss(logits:Tensor, targets:Tensor, label_smoothing:float=0.1) -> Tensor:
  # NOTE: fused sparse_categorical_crossentropy with label smoothing, returns mean loss scalar
  assert logits.dtype == dtypes.bfloat16, f"expected bf16, got {logits.dtype}"
  assert logits.ndim == 3, f"expected (MBS, SEQ, VOCAB), got {logits.shape}"
  MBS, SEQ, VOCAB = logits.shape
  rows = MBS * SEQ
  if isinstance(logits.device, tuple):
    axis = logits.uop.axis
    assert axis in (0, 1), f"unsupported sharding axis={axis} for CE loss"
    ndev = len(logits.device)
    loss_out = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0),
                      device=logits.device)
    max_out  = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0),
                      device=logits.device)
    lse_out  = Tensor(Tensor.invalids(rows // ndev, dtype=dtypes.float32, device=logits.device).uop.multi(0),
                      device=logits.device)
    local_shape = tuple(s//ndev if i == axis else s for i,s in enumerate(logits.shape))
    rows_per_dev = local_shape[0] * local_shape[1]
    seq_per_dev = local_shape[1]
  else:
    loss_out = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    max_out  = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    lse_out  = Tensor.invalids(rows, dtype=dtypes.float32, device=logits.device)
    rows_per_dev = rows
    seq_per_dev = SEQ
  targets_flat = targets.reshape(-1).cast(dtypes.int32)
  fxn = functools.partial(_custom_fused_ce_loss_fwd, vocab=VOCAB, rows=rows_per_dev, seq=seq_per_dev,
                          label_smoothing=label_smoothing)
  loss_out, max_out, lse_out, *_ = Tensor.custom_kernel(
    loss_out, max_out, lse_out, logits, targets_flat,
    fxn=fxn, grad_fxn=functools.partial(_fused_ce_loss_bwd, label_smoothing=label_smoothing))
  return loss_out.mean()
