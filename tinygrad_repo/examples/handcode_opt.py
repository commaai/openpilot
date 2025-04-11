from extra.models.resnet import ResNet50
from extra.mcts_search import mcts_search
from examples.mlperf.helpers import get_mlperf_bert_model
from tinygrad import Tensor, Device, dtypes, nn
from tinygrad.codegen.kernel import Kernel
from tinygrad.ops import Ops, sym_infer
from tinygrad.device import Compiled
from tinygrad.engine.search import beam_search, bufs_from_lin
from tinygrad.helpers import DEBUG, ansilen, getenv, colored, TRACEMETA
from extra.optimization.helpers import time_linearizer

def get_sched_resnet():
  mdl = ResNet50()
  optim = (nn.optim.LARS if getenv("LARS") else nn.optim.SGD)(nn.state.get_parameters(mdl))
  BS = getenv("BS", 64)

  # run model twice to get only what changes, these are the kernels of the model
  for _ in range(2):
    out = mdl(Tensor.empty(BS, 3, 224, 224))
    targets = [out]
    if getenv("BACKWARD"):
      optim.zero_grad()
      out.sparse_categorical_crossentropy(Tensor.empty(BS, dtype=dtypes.int)).backward()
      targets += [x for x in optim.schedule_step()]
    sched = Tensor.schedule(*targets)
    print(f"schedule length {len(sched)}")
  return sched

def get_sched_bert():
  mdl = get_mlperf_bert_model()
  optim = nn.optim.LAMB(nn.state.get_parameters(mdl))

  # fake data
  BS = getenv("BS", 9)
  input_ids = Tensor.empty((BS, 512), dtype=dtypes.float32)
  segment_ids = Tensor.empty((BS, 512), dtype=dtypes.float32)
  attention_mask = Tensor.empty((BS, 512), dtype=dtypes.default_float)
  masked_positions = Tensor.empty((BS, 76), dtype=dtypes.float32)
  masked_lm_ids = Tensor.empty((BS, 76), dtype=dtypes.float32)
  masked_lm_weights = Tensor.empty((BS, 76), dtype=dtypes.float32)
  next_sentence_labels = Tensor.empty((BS, 1), dtype=dtypes.float32)

  # run model twice to get only what changes, these are the kernels of the model
  for _ in range(2):
    lm_logits, seq_relationship_logits = mdl(input_ids, attention_mask, masked_positions, segment_ids)
    targets = [lm_logits, seq_relationship_logits]
    if getenv("BACKWARD"):
      optim.zero_grad()
      loss = mdl.loss(lm_logits, seq_relationship_logits, masked_lm_ids, masked_lm_weights, next_sentence_labels)
      # ignore grad norm and loss scaler for now
      loss.backward()
      targets += [x for x in optim.schedule_step()]
    sched = Tensor.schedule(*targets)
    print(f"schedule length {len(sched)}")
  return sched

if __name__ == "__main__":
  if getenv("HALF", 1):
    dtypes.default_float = dtypes.half

  # the device we are optimizing for
  device: Compiled = Device[Device.DEFAULT]
  if getenv("BACKWARD"): Tensor.training = True
  print(f"optimizing for {Device.DEFAULT}")

  sched = globals()[f"get_sched_{getenv('MODEL', 'resnet')}"]()
  sched = [x for x in sched if x.ast.op is Ops.SINK]

  # focus on one kernel
  if getenv("KERNEL", -1) >= 0: sched = sched[getenv("KERNEL", -1):getenv("KERNEL", -1)+1]

  # work with the schedule
  total_tm = 0
  running_gflops = 0
  usage = {}
  for i,si in enumerate(sched):
    if DEBUG >= 3: print(si.ast)

    rawbufs = bufs_from_lin(Kernel(si.ast))

    # "linearize" the op into uops in different ways
    lins: list[tuple[Kernel, str]] = []

    # always try hand coded opt
    lin = Kernel(si.ast, opts=device.renderer)
    lin.hand_coded_optimizations()
    lins.append((lin, "HC"))

    # maybe try tensor cores
    lin = Kernel(si.ast, opts=device.renderer)
    if lin.apply_tensor_cores():
      lins.append((lin, "TC"))

    # try a beam search
    if beam:=getenv("BEAM"):
      lin = Kernel(si.ast, opts=device.renderer)
      lin = beam_search(lin, rawbufs, beam, bool(getenv("BEAM_ESTIMATE", 1)))
      lins.append((lin, "BEAM"))

    # try MCTS
    if mcts:=getenv("MCTS"):
      lin = Kernel(si.ast, opts=device.renderer)
      lin = mcts_search(lin, rawbufs, mcts)
      lins.append((lin, "MCTS"))

    # benchmark the programs
    choices = []
    for lin, nm in lins:
      tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10, disable_cache=True)
      ops = (prg:=lin.to_program()).estimates.ops
      gflops = sym_infer(ops, {k:k.min for k in lin.ast.variables()})*1e-9/tm
      choices.append((tm, gflops, lin, prg, nm))

    sorted_choices = sorted(choices, key=lambda x: x[0])
    if DEBUG >= 1: # print all kernels
      for tm, gflops, lin, prg, nm in choices:
        print(f"                 kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(prg.global_size):18s} {str(prg.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS -- {colored(nm, 'green') if lin is sorted_choices[0][2] else nm}")

    tm, gflops, lin, prg, nm = sorted_choices[0]
    if getenv("SRC"):
      print(si.ast)
      print(lin.applied_opts)
      print(lin.to_program().src)
    total_tm += tm
    running_gflops += gflops * tm
    if (key := str([str(m) for m in si.metadata])) not in usage: usage[key] = (0, 0)
    usage[key] = (usage[key][0] + tm, usage[key][1] + 1)
    print(f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.name+' '*(37-ansilen(lin.name))} {str(prg.global_size):18s} {str(prg.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS {[repr(m) if TRACEMETA >= 2 else str(m) for m in si.metadata]}")
  print(f"******* total {total_tm*1000:.2f} ms, {running_gflops/total_tm:6.0f} GFLOPS")
  print("usage:")
  for k in sorted(usage, key=lambda x: -usage[x][0])[:10]:
    print(f"{usage[k][0]*1000:.2f} ms: {k} ({usage[k][1]} times)")
