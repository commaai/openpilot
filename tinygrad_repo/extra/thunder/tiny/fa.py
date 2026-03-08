import math

from tinygrad import Tensor, dtypes
from tinygrad.helpers import DEBUG
from tinygrad.uop.ops import UOp, Ops

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import GL, TileLayout

NUM_WORKERS = 1
Q_BLOCK_SIZE = 32
KV_BLOCK_SIZE = 32

def _sharded_empty(shape:Tensor, ref:Tensor, axis:int|None) -> Tensor:
  if not isinstance(ref.device, tuple): return Tensor.empty(*shape, dtype=ref.dtype, device=ref.device)
  shape = tuple(s // len(ref.device) if i == ref.uop.axis else s for i, s in enumerate(shape))
  axis = ref.uop.axis if axis is None else axis
  return Tensor(Tensor.empty(*shape, dtype=ref.dtype, device=ref.device).uop.multi(axis), dtype=ref.dtype, device=ref.device)

def _sharded_empty_like(ref:Tensor, axis:int|None=None) -> Tensor:
  return _sharded_empty(ref.shape, ref, axis)

def flash_attention(xq, xk, xv, attn_mask:Tensor|None=None, is_causal:bool=False):
  if len(xq.shape) == 3: xq, xk, xv = xq.unsqueeze(0), xk.unsqueeze(0), xv.unsqueeze(0)

  odtype = xq.dtype
  xq, xk, xv = xq.transpose(1, 2).cast(dtypes.bfloat16), xk.transpose(1, 2).cast(dtypes.bfloat16), xv.transpose(1, 2).cast(dtypes.bfloat16)

  _, N_, _, D_ = xq.shape
  block_size = max(Q_BLOCK_SIZE, KV_BLOCK_SIZE)
  assert D_ % block_size == 0, f"embedding dimension must be multiple of block size, got {D_=} {block_size=}"

  # pad to multiple of block size
  xq = xq.pad(((0, 0), (0, (block_size - (xq.shape[1] % block_size)) % block_size), (0, 0), (0, 0)))
  xk = xk.pad(((0, 0), (0, (block_size - (xk.shape[1] % block_size)) % block_size), (0, 0), (0, 0)))
  xv = xv.pad(((0, 0), (0, (block_size - (xv.shape[1] % block_size)) % block_size), (0, 0), (0, 0)))

  B, N, H, D = xq.shape
  H_KV = xk.shape[2]
  GROUP_SIZE = H // H_KV
  num_devices = len(xq.device) if isinstance(xq.device, tuple) else 1
  B_local = B // num_devices
  if DEBUG >= 2: print(f"Flash Attention {B=} {B_local=} {N=} {H=} {D=} {H_KV=} {GROUP_SIZE=}")

  def _custom_forward_impl(ou:UOp, l_vecu:UOp, qu:UOp, ku:UOp, vu:UOp, masku:UOp|None) -> UOp:
    with Kernel("fa_custom_forward", (H, N // (Q_BLOCK_SIZE*NUM_WORKERS), B_local), NUM_WORKERS * WARP_THREADS) as ker:
      warp = ker.warp

      o, q, k, v, l_vec = GL(ou, ker), GL(qu, ker), GL(ku, ker), GL(vu, ker), GL(l_vecu, ker)
      mask = GL(masku, ker) if masku is not None else None

      head = ker.blockIdx_x
      head_kv = head // GROUP_SIZE
      batch = ker.blockIdx_z
      q_seq = ker.blockIdx_y * NUM_WORKERS + ker.warpid

      k_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      v_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)

      q_reg_fl = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      q_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      q_reg_transposed = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      k_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      k_reg_transposed = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      v_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16, TileLayout.COL)
      o_reg = ker.rt((D, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      o_reg_transposed = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      att_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      att_block_mma = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      mask_reg = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.float32)
      mask_reg_transposed = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      max_vec_last = ker.rv(Q_BLOCK_SIZE, dtypes.float32)
      max_vec = ker.rv(Q_BLOCK_SIZE, dtypes.float32)
      norm_vec = ker.rv(Q_BLOCK_SIZE, dtypes.float32)
      scale_vec = ker.rv(Q_BLOCK_SIZE, dtypes.float32)

      max_vec = warp.neg_inf(max_vec)
      norm_vec = warp.zero(norm_vec)
      o_reg = warp.zero(o_reg)
      scale_vec = warp.ones(scale_vec)

      # load q tile
      q_reg_fl = warp.load(q_reg_fl, q, (), (batch, q_seq, head, 0), axis=1)
      q_reg_fl *= (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
      q_reg = warp.copy(q_reg, q_reg_fl)
      q_reg_transposed = warp.transpose(q_reg_transposed, q_reg)

      num_kv_blocks = (q_seq + 1) if is_causal else (N // KV_BLOCK_SIZE)
      for kv_idx in ker.range(num_kv_blocks):
        k_smem = warp.load(k_smem, k, (), (batch, kv_idx, head_kv, 0), axis=1)
        v_smem = warp.load(v_smem, v, (), (batch, kv_idx, head_kv, 0), axis=1)

        k_reg = warp.load(k_reg, k_smem)
        v_reg = warp.load(v_reg, v_smem)

        # mma qk^t
        att_block = warp.zero(att_block.after(kv_idx))
        k_reg_transposed = warp.transpose(k_reg_transposed, k_reg)
        att_block = warp.mma_AtB(att_block, k_reg_transposed, q_reg_transposed)

        # apply attention mask
        if is_causal:
          bs_rows, bs_cols, bs_stride = att_block.base_shape.rows, att_block.base_shape.cols, att_block.base_shape.stride
          q_base = q_seq * Q_BLOCK_SIZE + (warp.laneid % bs_cols)
          kv_base = kv_idx * KV_BLOCK_SIZE + (warp.laneid // bs_cols) * bs_stride
          att_block = warp.map(att_block,
            lambda x, idx: ((kv_base + idx[0]*bs_rows + idx[2]) > (q_base + idx[1]*bs_cols)).alu(Ops.WHERE, UOp.ufix(x._uop, -math.inf), x))
        elif mask is not None:
          mask_reg = warp.load(mask_reg, mask, (), (batch, 0, q_seq, kv_idx), axis=2)
          mask_reg_transposed = warp.transpose(mask_reg_transposed, mask_reg)
          att_block += mask_reg_transposed

        # softmax
        max_vec_last = warp.copy(max_vec_last.after(kv_idx), max_vec)
        max_vec = warp.col_reduce(max_vec.after(max_vec_last), att_block, lambda a, b: a.maximum(b), init_value=-math.inf)

        scale_vec = warp.map(scale_vec.after(max_vec_last, max_vec), lambda _, idx: max_vec_last[*idx] - max_vec[*idx])
        scale_vec = scale_vec.exp2()

        o_reg *= scale_vec
        norm_vec *= scale_vec

        att_block -= max_vec
        att_block = att_block.exp2()

        norm_vec = warp.col_reduce(norm_vec.after(scale_vec), att_block, lambda a, b: a + b)

        # mma av
        att_block_mma = warp.copy(att_block_mma.after(kv_idx, norm_vec), att_block)
        o_reg = warp.mma_AtB(o_reg, v_reg, att_block_mma)
      o_reg = ker.endrange()
      norm_vec = norm_vec.after(o_reg)
      max_vec = max_vec.after(o_reg)

      o_reg /= norm_vec

      o_reg_transposed = warp.transpose(o_reg_transposed, o_reg)
      o = warp.store(o, o_reg_transposed, (batch, q_seq, head, 0), (), axis=1)

      norm_vec = norm_vec.after(o)
      max_vec = max_vec.after(o)

      max_vec *= math.log(2)
      norm_vec = norm_vec.log2() * math.log(2)
      norm_vec += max_vec
      l_vec = warp.store(l_vec, norm_vec, (batch, head, 0, q_seq), (), axis=2)
      o = o.after(l_vec)

      return ker.finish()

  def custom_forward_causal(ou:UOp, l_vecu:UOp, qu:UOp, ku:UOp, vu:UOp) -> UOp:
    return _custom_forward_impl(ou, l_vecu, qu, ku, vu, None)

  def custom_forward_masked(ou:UOp, l_vecu:UOp, qu:UOp, ku:UOp, vu:UOp, masku:UOp) -> UOp:
    return _custom_forward_impl(ou, l_vecu, qu, ku, vu, masku)

  def _custom_backward_q_impl(dqu:UOp, dou:UOp, qu:UOp, ku:UOp, vu:UOp, masku:UOp|None, l_vecu:UOp, delta_vecu:UOp) -> UOp:
    with Kernel("fa_custom_backward_q", (H, N // (Q_BLOCK_SIZE*NUM_WORKERS), B_local), NUM_WORKERS * WARP_THREADS) as ker:
      warp = ker.warp

      dq, do, q, k, v = GL(dqu, ker), GL(dou, ker), GL(qu, ker), GL(ku, ker), GL(vu, ker)
      mask = GL(masku, ker) if masku is not None else None
      l_vec, delta_vec = GL(l_vecu, ker), GL(delta_vecu, ker)

      head = ker.blockIdx_x
      head_kv = head // GROUP_SIZE
      batch = ker.blockIdx_z
      q_seq = ker.blockIdx_y * NUM_WORKERS + ker.warpid

      k_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      v_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)

      q_reg_fl = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      q_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      q_reg_t = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      k_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      k_reg_t = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      k_reg_col = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16, TileLayout.COL)
      k_reg_col_t = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16)
      v_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      mask_reg = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.float32)
      mask_reg_transposed = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      dq_reg = ker.rt((D, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      dq_reg_transposed = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      do_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)

      dp_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      att_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      att_block_mma = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)

      l_vec_reg = ker.rv(Q_BLOCK_SIZE, dtypes.float32)
      delta_vec_reg = ker.rv(Q_BLOCK_SIZE, dtypes.float32)

      dq_reg = warp.zero(dq_reg)

      # load q tile
      q_reg_fl = warp.load(q_reg_fl, q, (), (batch, q_seq, head, 0), axis=1)
      q_reg_fl *= (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
      q_reg = warp.copy(q_reg, q_reg_fl)
      q_reg_t = warp.transpose(q_reg_t, q_reg)

      # load do tile
      do_reg = warp.load(do_reg, do, (), (batch, q_seq, head, 0), axis=1)

      # load l_vec
      l_vec_reg = warp.load(l_vec_reg, l_vec, (), (batch, head, 0, q_seq), axis=2)
      l_vec_reg *= 1.0 / math.log(2)
      delta_vec_reg = warp.load(delta_vec_reg, delta_vec, (), (batch, head, 0, q_seq), axis=2)

      num_kv_blocks = (q_seq + 1) if is_causal else (N // KV_BLOCK_SIZE)
      for kv_idx in ker.range(num_kv_blocks):
        k_smem = warp.load(k_smem, k, (), (batch, kv_idx, head_kv, 0), axis=1)
        v_smem = warp.load(v_smem, v, (), (batch, kv_idx, head_kv, 0), axis=1)

        k_reg = warp.load(k_reg, k_smem)
        k_reg_t = warp.transpose(k_reg_t, k_reg)
        k_reg_col = warp.load(k_reg_col, k_smem)
        k_reg_col_t = warp.transpose(k_reg_col_t, k_reg_col)
        v_reg = warp.load(v_reg, v_smem)

        # mma qk^t
        att_block = warp.zero(att_block.after(kv_idx))
        att_block = warp.mma_AtB(att_block, k_reg_t, q_reg_t)

        # apply attention mask
        if is_causal:
          bs_rows, bs_cols, bs_stride = att_block.base_shape.rows, att_block.base_shape.cols, att_block.base_shape.stride
          q_base = q_seq * Q_BLOCK_SIZE + (warp.laneid % bs_cols)
          kv_base = kv_idx * KV_BLOCK_SIZE + (warp.laneid // bs_cols) * bs_stride
          att_block = warp.map(att_block,
            lambda x, idx: ((kv_base + idx[0]*bs_rows + idx[2]) > (q_base + idx[1]*bs_cols)).alu(Ops.WHERE, UOp.ufix(x._uop, -math.inf), x))
        elif mask is not None:
          mask_reg = warp.load(mask_reg, mask, (), (batch, 0, q_seq, kv_idx), axis=2)
          mask_reg_transposed = warp.transpose(mask_reg_transposed, mask_reg)
          att_block += mask_reg_transposed

        att_block -= l_vec_reg
        att_block = att_block.exp2()

        dp_block = warp.zero(dp_block.after(kv_idx, att_block))
        dp_block = warp.mma_ABt(dp_block, v_reg, do_reg)
        dp_block -= delta_vec_reg
        att_block *= dp_block

        att_block *= 1.0 / math.sqrt(D)
        att_block_mma = warp.copy(att_block_mma, att_block)
        dq_reg = warp.mma_AB(dq_reg, k_reg_col_t, att_block_mma)
      dq_reg = ker.endrange()

      dq_reg_transposed = warp.transpose(dq_reg_transposed, dq_reg)
      dq = warp.store(dq, dq_reg_transposed, (batch, q_seq, head, 0), axis=1)

      return ker.finish()

  def custom_backward_q_causal(dqu:UOp, dou:UOp, qu:UOp, ku:UOp, vu:UOp, l_vecu:UOp, delta_vecu:UOp) -> UOp:
    return _custom_backward_q_impl(dqu, dou, qu, ku, vu, None, l_vecu, delta_vecu)

  def custom_backward_q_masked(dqu:UOp, dou:UOp, qu:UOp, ku:UOp, vu:UOp, masku:UOp, l_vecu:UOp, delta_vecu:UOp) -> UOp:
    return _custom_backward_q_impl(dqu, dou, qu, ku, vu, masku, l_vecu, delta_vecu)

  def _custom_backward_kv_impl(dku:UOp, dvu:UOp, dou:UOp, qu:UOp, ku:UOp, vu:UOp, masku:UOp|None, l_vecu:UOp, delta_vecu:UOp):
    with Kernel("fa_custom_backward_kv", (H_KV, N // (KV_BLOCK_SIZE*NUM_WORKERS), B_local), NUM_WORKERS * WARP_THREADS) as ker:
      warp = ker.warp

      dk, dv, do, q, k, v = GL(dku, ker), GL(dvu, ker), GL(dou, ker), GL(qu, ker), GL(ku, ker), GL(vu, ker)
      mask = GL(masku, ker) if masku is not None else None
      l_vec, delta_vec = GL(l_vecu, ker), GL(delta_vecu, ker)

      head_kv = ker.blockIdx_x
      batch = ker.blockIdx_z
      kv_seq = ker.blockIdx_y * NUM_WORKERS + ker.warpid

      q_smem = ker.st((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      do_smem = ker.st((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      att_smem = ker.st((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.bfloat16)

      q_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      q_reg_t = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      q_reg_col = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16, TileLayout.COL)
      k_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      k_reg_t = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      v_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      mask_reg = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.float32)
      mask_reg_transposed = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      dk_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.float32, TileLayout.COL)
      dv_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.float32, TileLayout.COL)
      do_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      do_reg_col = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16, TileLayout.COL)

      dp_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      att_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      att_block_mma = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      att_block_transposed = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      att_block_row = ker.rt((Q_BLOCK_SIZE, KV_BLOCK_SIZE), dtypes.bfloat16)

      l_vec_reg = ker.rv(Q_BLOCK_SIZE, dtypes.float32)
      delta_vec_reg = ker.rv(Q_BLOCK_SIZE, dtypes.float32)

      dk_reg = warp.zero(dk_reg)
      dv_reg = warp.zero(dv_reg)

      # load kv tile
      k_reg = warp.load(k_reg, k, (), (batch, kv_seq, head_kv, 0), axis=1)
      k_reg_t = warp.transpose(k_reg_t, k_reg)
      v_reg = warp.load(v_reg, v, (), (batch, kv_seq, head_kv, 0), axis=1)

      for q_idx in ker.range(N // Q_BLOCK_SIZE):
        for g in ker.range(GROUP_SIZE):
          head_q = head_kv * GROUP_SIZE + g

          # load q and do
          q_smem = warp.load(q_smem, q, (), (batch, q_idx, head_q, 0), axis=1)
          do_smem = warp.load(do_smem, do, (), (batch, q_idx, head_q, 0), axis=1)

          q_reg = warp.load(q_reg, q_smem)
          q_reg_t = warp.transpose(q_reg_t, q_reg)
          q_reg_col = warp.load(q_reg_col, q_smem)
          do_reg = warp.load(do_reg, do_smem)
          do_reg_col = warp.load(do_reg_col, do_smem)

          # load l_vec and delta_vec
          l_vec_reg = warp.load(l_vec_reg, l_vec, (), (batch, head_q, 0, q_idx), axis=2)
          l_vec_reg *= 1.0 / math.log(2)
          delta_vec_reg = warp.load(delta_vec_reg, delta_vec, (), (batch, head_q, 0, q_idx), axis=2)

          # mma qk^t
          att_block = warp.zero(att_block.after(g))
          att_block = warp.mma_AtB(att_block, k_reg_t, q_reg_t)
          att_block *= (1.0 / math.sqrt(D)) * (1.0 / math.log(2))

          # apply attention mask
          if is_causal:
            bs_rows, bs_cols, bs_stride = att_block.base_shape.rows, att_block.base_shape.cols, att_block.base_shape.stride
            q_base = q_idx * Q_BLOCK_SIZE + (warp.laneid % bs_cols)
            kv_base = kv_seq * KV_BLOCK_SIZE + (warp.laneid // bs_cols) * bs_stride
            att_block = warp.map(att_block,
              lambda x, idx: ((kv_base + idx[0]*bs_rows + idx[2]) > (q_base + idx[1]*bs_cols)).alu(Ops.WHERE, UOp.ufix(x._uop, -math.inf), x))
          elif mask is not None:
            mask_reg = warp.load(mask_reg, mask, (), (batch, 0, q_idx, kv_seq), axis=2)
            mask_reg_transposed = warp.transpose(mask_reg_transposed, mask_reg)
            att_block += mask_reg_transposed

          att_block -= l_vec_reg
          att_block = att_block.exp2()

          att_block_mma = warp.copy(att_block_mma, att_block)
          att_block_transposed = warp.transpose(att_block_transposed, att_block_mma)
          att_smem = warp.store(att_smem, att_block_transposed)
          att_block_row = warp.load(att_block_row, att_smem)
          dv_reg_ = warp.mma_AtB(dv_reg, att_block_row, do_reg_col)

          dp_block = warp.zero(dp_block.after(g, q_idx, dv_reg_))
          dp_block = warp.mma_ABt(dp_block, v_reg, do_reg)
          dp_block -= delta_vec_reg
          att_block *= dp_block

          att_block *= 1.0 / math.sqrt(D)
          att_block_mma = warp.copy(att_block_mma, att_block)
          att_block_transposed = warp.transpose(att_block_transposed, att_block_mma)
          att_smem = warp.store(att_smem, att_block_transposed)
          att_block_row = warp.load(att_block_row, att_smem)
          dk_reg = warp.mma_AtB(dk_reg, att_block_row, q_reg_col)
      dk_reg = ker.endrange(2)
      dv_reg = dv_reg.after(dk_reg)

      dv_reg = warp.map(dv_reg, lambda x, idx: x + v_reg[*idx].cast(dtypes.float32) * 1e-30)

      dk = warp.store(dk, dk_reg, (batch, kv_seq, head_kv, 0), axis=1)
      dv = warp.store(dv, dv_reg, (batch, kv_seq, head_kv, 0), axis=1)

      return ker.finish(2)

  def custom_backward_kv_causal(dku:UOp, dvu:UOp, dou:UOp, qu:UOp, ku:UOp, vu:UOp, l_vecu:UOp, delta_vecu:UOp):
    return _custom_backward_kv_impl(dku, dvu, dou, qu, ku, vu, None, l_vecu, delta_vecu)

  def custom_backward_kv_masked(dku:UOp, dvu:UOp, dou:UOp, qu:UOp, ku:UOp, vu:UOp, masku:UOp, l_vecu:UOp, delta_vecu:UOp):
    return _custom_backward_kv_impl(dku, dvu, dou, qu, ku, vu, masku, l_vecu, delta_vecu)

  single_device = xq.device[0] if isinstance(xq.device, tuple) else xq.device

  if is_causal:
    if attn_mask is not None: raise RuntimeError("cannot set attn_mask when is_causal=True")
  elif attn_mask is not None:
    if attn_mask.dtype == dtypes.bool: attn_mask = attn_mask.where(0, -float("inf"))
    if attn_mask.shape != (B, 1, N, N):
      attn_mask = attn_mask.expand(B, 1, N, N)
    if isinstance(xq.device, tuple) and not isinstance(attn_mask.device, tuple):
      attn_mask = attn_mask.shard(xq.device, axis=0)
  else:
    attn_mask = Tensor.zeros((B, 1, N, N), requires_grad=False, device=single_device, dtype=dtypes.float32)
    if isinstance(xq.device, tuple):
      attn_mask = attn_mask.shard(xq.device, axis=0)

  attn = _sharded_empty_like(xq, axis=0)
  l_vec = _sharded_empty((B, H, 1, N), xq, axis=0)

  def grad_causal(gradu:UOp, _) -> tuple[None, None, UOp, UOp, UOp]:
    grad = Tensor(gradu, device=gradu.device)
    grad_q = _sharded_empty_like(xq, axis=0)
    grad_k = _sharded_empty_like(xk, axis=0)
    grad_v = _sharded_empty_like(xv, axis=0)

    delta_vec = (grad * attn).sum(-1, dtype=dtypes.float32).transpose(1, 2).unsqueeze(-2).detach()

    grad_q = Tensor.custom_kernel(grad_q, grad, xq, xk, xv, l_vec, delta_vec, fxn=custom_backward_q_causal)[0]
    grad_k, grad_v = Tensor.custom_kernel(grad_k, grad_v, grad, xq, xk, xv, l_vec, delta_vec, fxn=custom_backward_kv_causal)[:2]
    return (None, None, grad_q.uop, grad_k.uop, grad_v.uop)

  def grad_masked(gradu:UOp, _) -> tuple[None, None, UOp, UOp, UOp, None]:
    grad = Tensor(gradu, device=gradu.device)
    grad_q = _sharded_empty_like(xq, axis=0)
    grad_k = _sharded_empty_like(xk, axis=0)
    grad_v = _sharded_empty_like(xv, axis=0)

    delta_vec = (grad * attn).sum(-1, dtype=dtypes.float32).transpose(1, 2).unsqueeze(-2).detach()

    grad_q = Tensor.custom_kernel(grad_q, grad, xq, xk, xv, attn_mask, l_vec, delta_vec, fxn=custom_backward_q_masked)[0]
    grad_k, grad_v = Tensor.custom_kernel(grad_k, grad_v, grad, xq, xk, xv, attn_mask, l_vec, delta_vec, fxn=custom_backward_kv_masked)[:2]
    return (None, None, grad_q.uop, grad_k.uop, grad_v.uop, None)

  if is_causal:
    attn, l_vec = Tensor.custom_kernel(attn, l_vec, xq, xk, xv, fxn=custom_forward_causal, grad_fxn=grad_causal)[:2]
  else:
    attn, l_vec = Tensor.custom_kernel(attn, l_vec, xq, xk, xv, attn_mask, fxn=custom_forward_masked, grad_fxn=grad_masked)[:2]
  attn_ = attn[:, :N_, :, :D_]

  return attn_.transpose(1, 2).cast(odtype)
