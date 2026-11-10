from functools import lru_cache

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from tinygrad.runtime.support.compiler_cuda import NVPTXCompiler
from tinygrad.uop.ops import KernelInfo, Ops, UOp


@triton.jit
def alpamayo_attention(O, L, Q, K, V, Mask, B: tl.constexpr, H: tl.constexpr, HK: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
                       D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr, CAUSAL: tl.constexpr, MASK: tl.constexpr,
                       QM: tl.constexpr, KN: tl.constexpr):
  pid = tl.program_id(0)
  split, block = pid % S, pid // S
  tile, bh = block % tl.cdiv(NQ, QM), block // tl.cdiv(NQ, QM)
  batch, head = bh // H, bh % H
  kh = head // (H // HK)
  rows, d = tile*QM + tl.arange(0, QM), tl.arange(0, DP)
  q = tl.load(Q + (bh*NQ + rows[:, None])*D + d[None, :], (rows[:, None] < NQ) & (d[None, :] < D), 0)
  maximum = tl.full((QM,), -float('inf'), tl.float32)
  sums, acc = tl.zeros((QM,), tl.float32), tl.zeros((QM, DP), tl.float32)
  span = tl.cdiv(NK, S*KN)*KN
  end = tl.minimum(NK, (split+1)*span)
  if CAUSAL:
    end = tl.minimum(end, (tile+1)*QM)
  for start in range(split*span, end, KN):
    keys = start + tl.arange(0, KN)
    k = tl.load(K + ((batch*HK + kh)*NK + keys[None, :])*D + d[:, None], (keys[None, :] < NK) & (d[:, None] < D), 0)
    score = tl.dot(q, k) * (D**-0.5)
    score = tl.where(keys[None, :] < NK, score, -float('inf'))
    if CAUSAL:
      score = tl.where(keys[None, :] <= rows[:, None], score, -float('inf'))
    if MASK:
      score += tl.load(Mask + keys, keys < NK, -float('inf'))[None, :]
    new_max = tl.maximum(maximum, tl.max(score, 1))
    alpha = tl.exp(tl.where(new_max != -float('inf'), maximum-new_max, 0.))
    p = tl.exp(tl.where(score != -float('inf'), score-new_max[:, None], -float('inf')))
    sums = sums*alpha + tl.sum(p, 1)
    acc = acc*alpha[:, None]
    v = tl.load(V + ((batch*HK + kh)*NK + keys[:, None])*D + d[None, :], (keys[:, None] < NK) & (d[None, :] < D), 0)
    acc = tl.dot(p.to(tl.bfloat16), v, acc)
    maximum = new_max
  out = tl.where(sums[:, None] > 0, acc/sums[:, None], 0.)
  tl.store(O + ((split*B*H + bh)*NQ + rows[:, None])*D + d[None, :], out, (rows[:, None] < NQ) & (d[None, :] < D))
  tl.store(L + (split*B*H + bh)*NQ + rows, maximum + tl.log(sums), rows < NQ)


@lru_cache(None)
def compile_kernel(batch, heads, kv_heads, queries, keys, width, splits, causal, mask, tile, arch):
  constants = {'B': batch, 'H': heads, 'HK': kv_heads, 'NQ': queries, 'NK': keys, 'D': width, 'DP': triton.next_power_of_2(width),
               'S': splits, 'CAUSAL': causal, 'MASK': mask, 'QM': tile, 'KN': 64}
  signature = {'O': '*bf16' if splits == 1 else '*fp32', 'L': '*fp32', 'Q': '*bf16', 'K': '*bf16', 'V': '*bf16', 'Mask': '*fp32'}
  kernel = triton.compile(ASTSource(alpamayo_attention, signature, constants), target=GPUTarget('cuda', arch, 32),
                         options={'num_warps': 4, 'num_stages': 2})
  # tinygrad's native NV loader obtains shared memory size from the cubin's static section.
  declaration = '.extern .shared .align 16 .b8 global_smem[];'
  assert declaration in kernel.asm['ptx'] and kernel.metadata.global_scratch_size == 0
  ptx = kernel.asm['ptx'].replace(declaration,
                                 f'.shared .align 16 .b8 global_smem[{kernel.metadata.shared}];').replace(f'sm_{arch}a', f'sm_{arch}')
  return kernel.name, ptx, NVPTXCompiler(f'sm_{arch}').compile_cached(ptx)


def attention_program(*buffers, shape, arch, blocks):
  name, ptx, binary = compile_kernel(*shape, arch)
  value = sum(x.flatten()[0].load().float() for x in buffers[2:])
  sink = UOp.sink(*(x.flatten()[0].store(value) for x in buffers[:2]), UOp.special(blocks, 'gidx0'), UOp.special(128, 'lidx0'),
                  arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=ptx), UOp(Ops.BINARY, arg=binary)))
