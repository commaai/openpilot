import ctypes, pathlib, argparse, pickle, re, functools, dataclasses, itertools
from tinygrad.helpers import temp, unwrap, DEBUG
from tinygrad.device import ProfileEvent, ProfileDeviceEvent, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent, ProfilePMCEvent
from tinygrad.runtime.autogen import llvm, rocprof
from tinygrad.runtime.support.elf import elf_loader

# to pass NULL to callbacks
llvm.LLVMCreateDisasmCPUFeatures.argtypes = tuple(llvm.LLVMCreateDisasmCPUFeatures.argtypes[:5]) + (ctypes.c_void_p, ctypes.c_void_p)
def llvm_disasm(arch:str, lib:bytes) -> dict[int, tuple[str, int]]:
  llvm.LLVMInitializeAMDGPUTargetInfo()
  llvm.LLVMInitializeAMDGPUTargetMC()
  llvm.LLVMInitializeAMDGPUAsmParser()
  llvm.LLVMInitializeAMDGPUDisassembler()
  ctx = llvm.LLVMCreateDisasmCPUFeatures("amdgcn-amd-amdhsa".encode(), arch.encode(), "".encode(), None, 0, None, None)

  image, sections, relocs = elf_loader(lib)
  text = next((sh.header for sh in sections if sh.name == ".text"), None)
  off, sz = unwrap(text).sh_addr, unwrap(text).sh_size

  addr_table:dict[int, tuple[str, int]] = {}
  out = ctypes.create_string_buffer(128)
  cur_off = off
  while cur_off < sz + off:
    view = (ctypes.c_ubyte * ((sz + off) - cur_off)).from_buffer_copy(memoryview(image)[cur_off:])
    instr_sz = llvm.LLVMDisasmInstruction(ctx, view, ctypes.c_uint64(len(view)), ctypes.c_uint64(0), out, ctypes.c_size_t(128))
    addr_table[cur_off] = (out.value.decode("utf-8", "replace").strip(), instr_sz)
    cur_off += instr_sz
  return addr_table

@dataclasses.dataclass(frozen=True)
class InstExec:
  typ:str
  inst:str
  stall:int
  dur:int
  time:int

@dataclasses.dataclass(frozen=True)
class WaveExec:
  wave_id:int
  cu:int
  simd:int
  begin_time:int
  end_time:int
  insts:list[InstExec]

class _ROCParseCtx:
  def __init__(self, dev_evs:dict[str, ProfileDeviceEvent], sqtt_evs:list[ProfileSQTTEvent], prog_evs:list[ProfileProgramEvent]):
    self.dev_evs, self.sqtt_evs, self.prog_evs = dev_evs, iter(sqtt_evs), prog_evs
    self.disasms:dict[tuple[str, int], tuple[str, int]] = {}
    self.inst_execs:dict[str, list[WaveExec]] = {}

    for prog in prog_evs:
      arch = "gfx%d%x%x" % ((trgt:=unwrap(dev_evs[prog.device].props)['gfx_target_version']) // 10000, (trgt // 100) % 100, trgt % 100)
      for addr, info in llvm_disasm(arch, unwrap(prog.lib)).items():
        self.disasms[(prog.name, unwrap(prog.base) + addr)] = info

  def next_sqtt(self):
    x = next(self.sqtt_evs, None)
    self.active_kern = x.kern if x is not None else None
    self.active_se = x.se if x is not None else None
    self.active_blob = (ctypes.c_ubyte * len(x.blob)).from_buffer_copy(x.blob) if x is not None else None
    return self.active_blob

  def on_occupancy_ev(self, ev):
    if DEBUG >= 5: print("OCC", ev.time, self.active_se, ev.cu, ev.simd, ev.wave_id, ev.start)

  def on_wave_ev(self, ev):
    if DEBUG >= 5: print("WAVE", ev.wave_id, self.active_se, ev.cu, ev.simd, ev.contexts, ev.begin_time, ev.end_time)

    inst_execs:list[InstExec] = []
    for j in range(ev.instructions_size):
      inst_ev = ev.instructions_array[j]
      inst_typ = rocprof.rocprofiler_thread_trace_decoder_inst_category_t__enumvalues[inst_ev.category]
      inst_disasm = self.disasms[(unwrap(self.active_kern), unwrap(inst_ev.pc.address))][0]
      inst_execs.append(InstExec(inst_typ, inst_disasm, inst_ev.stall, inst_ev.duration, inst_ev.time))

    if ev.instructions_size > 0:
      self.inst_execs.setdefault(unwrap(self.active_kern), []).append(WaveExec(ev.wave_id, ev.cu, ev.simd, ev.begin_time, ev.end_time, inst_execs))

def decode(profile:list[ProfileEvent]) -> _ROCParseCtx:
  dev_events:dict[str, ProfileDeviceEvent] = {}
  sqtt_events:list[ProfileSQTTEvent] = []
  prog_events:list[ProfileProgramEvent] = []
  for e in profile:
    if isinstance(e, ProfileDeviceEvent): dev_events[e.device] = e
    if isinstance(e, ProfileSQTTEvent): sqtt_events.append(e)
    if isinstance(e, ProfileProgramEvent) and e.device.startswith("AMD"): prog_events.append(e)

  ROCParseCtx = _ROCParseCtx(dev_events, sqtt_events, prog_events)

  @rocprof.rocprof_trace_decoder_se_data_callback_t
  def copy_cb(buf, buf_size, data_ptr):
    if (prof_info:=ROCParseCtx.next_sqtt()) is None: return 0
    buf[0] = ctypes.cast(prof_info, ctypes.POINTER(ctypes.c_ubyte))
    buf_size[0] = len(prof_info)
    return len(prof_info)

  @rocprof.rocprof_trace_decoder_trace_callback_t
  def trace_cb(record_type, events_ptr, n, data_ptr):
    match record_type:
      case rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
        for ev in (rocprof.rocprofiler_thread_trace_decoder_occupancy_t * n).from_address(events_ptr): ROCParseCtx.on_occupancy_ev(ev)
      case rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
        for ev in (rocprof.rocprofiler_thread_trace_decoder_wave_t * n).from_address(events_ptr): ROCParseCtx.on_wave_ev(ev)
      case _:
        if DEBUG >= 5: print(rocprof.rocprofiler_thread_trace_decoder_record_type_t__enumvalues[record_type], events_ptr, n)
    return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

  @rocprof.rocprof_trace_decoder_isa_callback_t
  def isa_cb(instr_ptr, mem_size_ptr, size_ptr, pc, data_ptr):
    instr, mem_size_ptr[0] = ROCParseCtx.disasms[(unwrap(ROCParseCtx.active_kern), pc.address)]

    # this is the number of bytes to next instruction, set to 0 for end_pgm
    if instr == "s_endpgm": mem_size_ptr[0] = 0
    if (max_sz:=size_ptr[0]) == 0: return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES

    # truncate the instr if it doesn't fit
    if (str_sz:=len(instr_bytes:=instr.encode()))+1 > max_sz: str_sz = max_sz
    ctypes.memmove(instr_ptr, instr_bytes, str_sz)
    size_ptr[0] = str_sz

    return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

  try:
    rocprof.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, None)
  except AttributeError as e: raise RuntimeError("Failed to find rocprof-trace-decoder. Run sudo ./extra/sqtt/install_sqtt_decoder.py to install") from e
  return ROCParseCtx

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', type=pathlib.Path, help='Path to profile', default=pathlib.Path(temp("profile.pkl", append_user=True)))
  args = parser.parse_args()

  with args.profile.open("rb") as f: profile = pickle.load(f)
  rctx = decode(profile)
  print('SQTT:', rctx.inst_execs.keys())

  for ev in profile:
    if not isinstance(ev, ProfilePMCEvent): continue
    print(f"PMC Event: dev={ev.device} kern={ev.kern}")
    ptr = 0
    for s in ev.sched:
      view = memoryview(ev.blob).cast('Q')
      print(f"\t{s.name}")
      for xcc, inst, se_idx, sa_idx, wgp_idx in itertools.product(range(s.xcc), range(s.inst), range(s.se), range(s.sa), range(s.wgp)):
        print(f"\t\tXCC {xcc} Inst {inst} SE {se_idx} SA {sa_idx} WGP {wgp_idx}: {view[ptr]:#x}")
        ptr += 1
