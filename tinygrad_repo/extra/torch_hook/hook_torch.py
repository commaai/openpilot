import ctypes, struct, platform, pathlib, os, binascii, itertools
from hexdump import hexdump
from tinygrad.device import Device
from tinygrad import Tensor
from tinygrad.dtype import _from_torch_dtype
from tinygrad.helpers import to_mv, DEBUG, getenv, colored, time_to_str

import extra.torch_hook.hook_cuda as hook_cuda

# settings to profile gemm in the __main__ example: TINY_MIRROR=1;CUDA=1;RUN_ONLY=9
# nvprof sample command (this will sample all kernels):
# ncu --export ~/nvprof_data --force-overwrite --rule AchievedOccupancy --rule Compute --rule LaunchConfiguration --rule Memory --rule PMSamplingData --rule SOLBottleneck --rule TheoreticalOccupancy --rule WorkloadImbalance python3 extra/torch_hook/hook_torch.py
# or just run nsight compute from the host to the machine.

TINY_MIRROR = getenv("TINY_MIRROR", 1) # should mirror aten ops to tiny backend
RUN_ONLY = getenv("RUN_ONLY", -1) # run only a specific aten call
REALIZE = getenv("REALIZE", 1) # realize and wait each aten call
WRAP_TINY = getenv("WRAP_TINY", TINY_MIRROR) # reuse cuda tensors
FULL_KERN_NAME = getenv("FULL_KERN_NAME", 0) # print full kernel name

print("importing torch...")
import torch
print("importing torch done:", torch.__version__, torch.__file__)

if TINY_MIRROR:
  print("importing tiny torch")
  import extra.torch_backend.backend as tiny_torch
  print("importing tiny torch done")

torch.set_default_device("cuda")

cuda_to_tiny_mappings = {}

enumerator_aten_calls = itertools.count(0)
from torch.utils._python_dispatch import TorchDispatchMode
class DispatchLog(TorchDispatchMode):
  def __torch_dispatch__(self, func, types, args, kwargs=None):
    txt_args = []
    should_call_tiny = kwargs.get('device') is not None and kwargs['device'].type == "cuda"

    def can_print_arg(arg):
      return args is None or isinstance(arg, (str, int, float, bool))

    def create_tiny_mapping(arg):
      if WRAP_TINY:
        tt = Tensor.from_blob(arg.data_ptr(), arg.shape, dtype=_from_torch_dtype(arg.dtype))
        cuda_to_tiny_mappings[arg] = tiny_torch.wrap(tt)

    for i,arg in enumerate(args):
      if torch.is_tensor(arg):
        if arg.device.type == "cuda":
          should_call_tiny = True
          if WRAP_TINY: create_tiny_mapping(arg)
        txt_args.append(f"tensor({arg.shape} {arg.device} {arg.dtype})")
      elif can_print_arg(arg): txt_args.append(f'{arg}')
      else: txt_args.append(f"{type(arg)}")
    for k,v in (kwargs or {}).items():
      if torch.is_tensor(v):
        if arg.device.type == "cuda":
          should_call_tiny = True
          if WRAP_TINY: create_tiny_mapping(arg)
        txt_args.append(f"{k}:tensor({v.shape} {v.device} {v.dtype})")
      elif can_print_arg(arg): txt_args.append(f'{k}:{arg}"')
      else: txt_args.append(f"{type(arg)}")

    # magenta-colored kerenls mirrored to tiny backend.
    aten_id = next(enumerator_aten_calls)
    should_call_tiny = TINY_MIRROR and should_call_tiny
    print(colored(f"#{aten_id} {func}", "magenta" if should_call_tiny else "cyan") + "("+", ".join(txt_args)+")", flush=True)

    # ignore dispatches if needed
    hook_cuda.push_ignore_dispatch(RUN_ONLY >= 0 and RUN_ONLY != aten_id)
    orig_x = func(*args, **(kwargs or {}))

    def print_events(evs, name, out_addr):
      for ev in evs:
        if isinstance(ev, hook_cuda.HookKernelCallEvent):
          txt_params = []
          for param in ev.params:
            if isinstance(param, hook_cuda.HookTensorParamEvent):
              is_out = param.cuda_address == out_addr
              txt_params += [f"{'result ' if is_out else ''}Tensor{param.enum}({param.cuda_address:#x})"]

          just_kern_name = ev.name
          if not FULL_KERN_NAME:
            just_kern_name = ev.name.replace("(anonymous namespace)", "").replace("void ", "").split("<")[0].split("(")[0].split("::")[-1]
          print(f"\t {name} kernel {just_kern_name} {ev.grid} {ev.block} {ev.ptm}\n\t\t({', '.join(txt_params)})")
        else: print("\t", name, ev)

    if REALIZE:
      torch.cuda.synchronize()
      cuda_events = hook_cuda.collect_events(clear=True)
      print_events(cuda_events, colored("cuda", "cyan"), orig_x.data_ptr() if torch.is_tensor(orig_x) else 0x0)

    if should_call_tiny:
      # replace with tiny tensor
      tiny_args, tiny_kwargs = [], {}
      for arg in args:
        if torch.is_tensor(arg): tiny_args.append(cuda_to_tiny_mappings[arg])
        else: tiny_args.append(arg)

      for k,v in (kwargs or {}).items():
        if torch.is_tensor(v): tiny_kwargs[k] = cuda_to_tiny_mappings[v]
        else: tiny_kwargs[k] = v
      if 'device' in tiny_kwargs and kwargs['device'].type == "cuda":
        tiny_kwargs['device'] = torch.device("tiny")

      tiny_x = func(*tiny_args, **tiny_kwargs)

      # TODO: this is a hack, any way to do this better?
      if REALIZE:
        out_addr = 0x0
        _ = tiny_x.cpu().numpy()
        if torch.is_tensor(tiny_x) and tiny_x.device.type == "tiny":
          tt = tiny_torch.unwrap(tiny_x)
          try: out_addr = tt.uop.buffer._buf.value
          except Exception: pass
        tiny_events = hook_cuda.collect_events(clear=True)
        print_events(tiny_events, colored("tiny", "magenta"), out_addr)

      if not WRAP_TINY: cuda_to_tiny_mappings[orig_x] = tiny_x

    hook_cuda.pop_ignore_dispatch()
    return orig_x
DispatchLog().__enter__()

if __name__ == "__main__":
  if getenv("RESNET"):
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model = model.cuda()
    model.eval()

    if getenv("COMPILE"): model = torch.compile(model)

    X = torch.rand(getenv("BS", 1), 3, 288, 288, device='cuda')
    model(X)

    print("\n\n\n****** second run ******\n")
    model(X)
  else:
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    a += 1
    b += 2
    a = a.exp2()
    b = b.exp2()
    a += b
    c = a @ b
    print("tensor math done", c.cpu().numpy())
