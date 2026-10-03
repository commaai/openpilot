from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes, nn
from tinygrad.uop.ops import UOp, Ops, KernelInfo, sint
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like, compile_hip

VOCAB, EMBED = 128256, 2880
FWD_THREADS, ROWS_PER_WG = 512, 8

@functools.cache
def _custom_embedding_fwd(out:UOp, idx:UOp, weight:UOp) -> UOp:
  tokens = idx.numel()
  threads, workgroups = UOp.special(FWD_THREADS, "lidx0"), UOp.special(tokens // ROWS_PER_WG, "gidx0")
  sink = UOp.sink(out.base, idx.base, weight.base, threads, workgroups,
                  arg=KernelInfo(f"gptoss_embedding_fwd_{tokens}_{VOCAB}_{EMBED}_v16_t{FWD_THREADS}_nt",
                                 estimates=Estimates(mem=tokens*4 + tokens*EMBED*4)))
  src = (pathlib.Path(__file__).parent/"embedding_fwd.cpp").read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, []))))

def gptoss_embedding_fwd(weight:Tensor, idx:Tensor) -> Tensor:
  out_shape = idx.shape + (EMBED,)
  out = alloc_like(out_shape, dtypes.bfloat16, idx.device, idx.uop.axis).clone()
  out, *_ = Tensor.custom_kernel(out, idx.reshape(-1), weight, fxn=_custom_embedding_fwd)
  return out

THREADS = 256

@functools.cache
def _custom_init_heads(head:UOp) -> UOp:
  vocab = head.numel()
  threads, workgroups = UOp.special(THREADS, "lidx0"), UOp.special((vocab+THREADS-1)//THREADS, "gidx0")
  sink = UOp.sink(head.base, threads, workgroups,
                  arg=KernelInfo(f"embedding_bwd_init_heads_{vocab}", estimates=Estimates(mem=vocab*4)))
  src = (pathlib.Path(__file__).parent/"embedding_bwd.cpp").read_text()
  defines = [f"-DVOCAB={vocab}", f"-DTHREADS={THREADS}", "-DINIT_HEADS=1"]
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_build_links(next_idx:UOp, head:UOp, idx:UOp) -> UOp:
  tokens, vocab = idx.numel(), head.numel()
  threads, workgroups = UOp.special(THREADS, "lidx0"), UOp.special((tokens+THREADS-1)//THREADS, "gidx0")
  sink = UOp.sink(next_idx.base, head.base, idx.base, threads, workgroups,
                  arg=KernelInfo(f"embedding_bwd_build_links_{tokens}_{vocab}", estimates=Estimates(ops=tokens, mem=3*tokens*4)))
  src = (pathlib.Path(__file__).parent/"embedding_bwd.cpp").read_text()
  defines = [f"-DTOKENS={tokens}", f"-DVOCAB={vocab}", f"-DTHREADS={THREADS}", "-DBUILD_LINKS=1"]
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_reduce(out:UOp, grad_emb:UOp, head:UOp, next_idx:UOp) -> UOp:
  vocab, embed = out.shape
  tokens = next_idx.numel()
  threads = UOp.special(THREADS, "lidx0")
  workgroups = UOp.special(vocab*((embed+THREADS-1)//THREADS), "gidx0")
  sink = UOp.sink(out.base, grad_emb.base, head.base, next_idx.base, threads, workgroups,
                  arg=KernelInfo(f"embedding_bwd_owner_reduce_{tokens}_{vocab}_{embed}",
                                 estimates=Estimates(ops=tokens*embed, mem=tokens*embed*2+vocab*embed*2)))
  src = (pathlib.Path(__file__).parent/"embedding_bwd.cpp").read_text()
  defines = [f"-DTOKENS={tokens}", f"-DVOCAB={vocab}", f"-DEMBED={embed}", f"-DTHREADS={THREADS}"]
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def embedding_bwd_owner(grad_emb:Tensor, idx:Tensor, vocab:sint) -> Tensor:
  grad_emb = grad_emb.reshape(idx.numel(), grad_emb.shape[-1])
  device = grad_emb.device
  head = alloc_like((vocab,), dtypes.int32, device)
  next_idx = alloc_like((idx.numel(),), dtypes.int32, device)
  out = alloc_like((vocab, grad_emb.shape[-1]), dtypes.bfloat16, device)
  head, *_ = Tensor.custom_kernel(head, fxn=_custom_init_heads)
  next_idx, *_ = Tensor.custom_kernel(next_idx, head, idx.reshape(-1), fxn=_custom_build_links)
  out, *_ = Tensor.custom_kernel(out, grad_emb, head, next_idx, fxn=_custom_reduce)
  return out

@functools.cache
def _embedding_fwd_fxn(wp:UOp, ip:UOp, device:str|tuple[str, ...]) -> Tensor:
  return gptoss_embedding_fwd(Tensor(wp, device=device), Tensor(ip, device=device))

def _embedding_bwd(grad_emb:UOp, call:UOp) -> tuple:
  weight, idx = call.src[1:3]
  device = Tensor(weight).device
  def gather(u:UOp) -> Tensor:
    t = Tensor(u, device=device)
    if not isinstance(device, tuple) or (axis := t.uop.axis) is None: return t
    return Tensor.cat(*t.chunk(len(device), dim=axis), dim=axis).contiguous()
  return embedding_bwd_owner(gather(grad_emb), gather(idx), weight.shape[0]).uop, None

class GPTOSSEmbedding(nn.Embedding):
  def __call__(self, idx:Tensor) -> Tensor:
    fxn = _embedding_fwd_fxn(self.weight.as_param(0).uop, idx.as_param(1).uop, self.weight.device)
    return Tensor.call(self.weight, idx, fxn=fxn, grad_fxn=_embedding_bwd)
