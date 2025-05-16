#!/usr/bin/env python3
import os
if "NOOPT" not in os.environ: os.environ["NOOPT"] = "1"
from tinygrad import Device, nn, Tensor, dtypes, Variable
Device.DEFAULT = "CPU"
from train_gpt2 import GPT, GPTConfig
from tinygrad.helpers import dedup, to_function_name, flatten, getenv, GlobalCounters, ansilen, to_function_name
from tinygrad.engine.realize import get_kernel, run_schedule
from tinygrad.engine.memory import memory_planner
from tinygrad.ops import Ops

TIMING = getenv("TIMING")

if __name__ == "__main__":
  model = GPT(GPTConfig(n_layer=getenv("NLAYER", 12), n_head=12, n_embd=768))
  #model.load_pretrained()
  for p in nn.state.get_parameters(model): p.replace(Tensor.empty(p.shape, dtype=p.dtype)) # fake load pretrained

  #early_sched = create_schedule([x.lazydata for x in nn.state.get_parameters(model)])
  #print(f"built model {len(early_sched)}")

  #B, T = Variable("B", 1, 128).bind(4), 64 #Variable("T", 1, 1024).bind(64)
  B, T = 4, 64

  Tensor.training = True
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-4)
  warmup_count = getenv("WARMUP", 3)
  for i in range(warmup_count):  # TODO: why does it take three and not two to stabilize
    GlobalCounters.reset()
    X = Tensor.empty(4, 64, dtype=dtypes.int).reshape(B, T)
    Y = Tensor.empty(4, 64, dtype=dtypes.int).reshape(B, T)
    _, loss = model(X, Y)
    optimizer.zero_grad()
    if getenv("BACKWARD", 1):
      loss.backward()
      tensors = optimizer.schedule_step()
    else:
      tensors = []
    sched = loss.schedule(*tensors)
    print(f"calls {i}:", len(sched))
    #run_schedule(sched[:])
  sched = memory_planner(sched)
  ast_dedup = dedup([si.ast for si in sched if si.ast.op is Ops.SINK])
  srcs = {}
  for ast in ast_dedup:
    k = get_kernel(Device["CPU"].renderer, ast)
    k.linearize()
    src = Device["CPU"].renderer.render(to_function_name(k.name), k.uops)
    srcs[ast] = (k.name, src)
  print("functions:", len(srcs))
  used_buffers = dedup(flatten([si.bufs for si in sched]))
  numbered_bufs = {x:i for i,x in enumerate(used_buffers)}
  print("buffers:", len(numbered_bufs))

  state_dict = nn.state.get_state_dict(model)
  state_dict.update({'X': X, 'Y': Y, 'loss': loss})
  grad_state_dict = {}
  for k,v in state_dict.items():
    if v.lazydata.base.buffer not in used_buffers: print(f"UNUSED: {k}")
    if v.grad is not None: grad_state_dict['grad_'+k] = v.grad
  state_dict.update(grad_state_dict)
  state_dict.update({'adam_b1_t': optimizer.b1_t, 'adam_b2_t': optimizer.b2_t, 'adam_lr': optimizer.lr})
  inverse_state_dict = {v:k for k,v in state_dict.items()}
  for p,m,v in zip(optimizer.params, optimizer.m, optimizer.v):
    nm = inverse_state_dict[p]
    state_dict["adam_m_"+nm] = m
    state_dict["adam_v_"+nm] = v
  named_buffers = {v.lazydata.base.buffer:k.replace(".", "_") for k,v in state_dict.items()}

  c_code = ["#include <stdlib.h>", "#include <tgmath.h>", "#include <stdbool.h>"]
  if TIMING: c_code += ["#include <stdio.h>", "#include <time.h>"]
  c_code += [x[1].replace(" restrict ", " ")+"\n" for x in srcs.values()]

  premain = ["int main() {"]
  if TIMING:
    premain += ["  struct timespec tm0; clock_gettime(CLOCK_MONOTONIC, &tm0);"]
  lst = 0
  main = []

  all_bufs = []
  for i,si in enumerate(sched):
    bufs = [(named_buffers.get(b, f"b{numbered_bufs[b]}"), b) for b in si.bufs]
    all_bufs += bufs
    if si.ast.op is not Ops.SINK:
      print(f"// {si.ast.op}", bufs)
    else:
      print(f"{srcs[si.ast][0]}({', '.join([x[0] for x in bufs])})")
      main.append(f"  {to_function_name(srcs[si.ast][0])}({', '.join([x[0] for x in bufs])});")
      if TIMING:
        main.append(f"  struct timespec tm{i+1}; clock_gettime(CLOCK_MONOTONIC, &tm{i+1});")
        main.append(f"  printf(\"%10.2f ms + %7.2f ms @ {to_function_name(srcs[si.ast][0])}\\n\"," +\
                    f"((tm{i+1}.tv_sec-tm{0}.tv_sec) + (tm{i+1}.tv_nsec-tm{0}.tv_nsec) / 1e9) * 1e3," +\
                    f"((tm{i+1}.tv_sec-tm{lst}.tv_sec) + (tm{i+1}.tv_nsec-tm{lst}.tv_nsec) / 1e9) * 1e3);")
      lst = i+1
      #call = f"{srcs[si.ast][0]}({', '.join(bufs)})"
      #call += " "*(80-ansilen(call))
      #print(f"{call} // {i+1}")
      #print(srcs[si.ast][1])
  main.append("}")

  mallocs = [f"  {b.dtype.name}* {n} = ({b.dtype.name}*)malloc({b.nbytes});" for n,b in dedup(all_bufs)]

  with open("out.c", "w") as f: f.write('\n'.join(c_code+premain+mallocs+main))
