import ctypes, pathlib, argparse, pickle, re, functools, dataclasses, itertools, threading
from typing import Generator
from tinygrad.helpers import temp, unwrap, DEBUG
from tinygrad.device import ProfileEvent, ProfileDeviceEvent, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent, ProfilePMCEvent
from tinygrad.runtime.autogen import llvm, rocprof
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.viz.serve import llvm_disasm

@dataclasses.dataclass(frozen=True)
class InstExec:
  typ:str
  pc:int
  stall:int
  dur:int
  time:int

@dataclasses.dataclass(frozen=True)
class WaveSlot:
  wave_id:int
  cu:int
  simd:int
  se:int
  @property
  def cu_loc(self) -> str: return f"SE:{self.se} CU:{self.cu}"
  @property
  def simd_loc(self) -> str: return f"{self.cu_loc} SIMD:{self.simd}"
  @property
  def wave_loc(self) -> str: return f"{self.simd_loc} W:{self.wave_id}"

@dataclasses.dataclass(frozen=True)
class WaveExec(WaveSlot):
  begin_time:int
  end_time:int
  insts:bytearray
  def unpack_insts(self) -> Generator[InstExec, None, None]:
    sz = ctypes.sizeof(struct:=rocprof.rocprofiler_thread_trace_decoder_inst_t)
    insts_array = (struct*(len(self.insts)//sz)).from_buffer(self.insts)
    for inst in insts_array:
      inst_typ = rocprof.enum_rocprofiler_thread_trace_decoder_inst_category_t.get(inst.category)
      yield InstExec(inst_typ, inst.pc.address, inst.stall, inst.duration, inst.time)

@dataclasses.dataclass(frozen=True)
class OccEvent(WaveSlot):
  time:int
  start:int

RunKey = tuple[str, int]

class _ROCParseCtx:
  def __init__(self, sqtt_evs:list[ProfileSQTTEvent], disasms:dict[str, dict[int, tuple[str, int]]]):
    self.sqtt_evs, self.disasms = iter(sqtt_evs), disasms
    self.inst_execs:dict[RunKey, list[WaveExec]] = {}
    self.occ_events:dict[RunKey, list[OccEvent]] = {}

  def next_sqtt(self):
    x = next(self.sqtt_evs, None)
    self.active_run = (x.kern, x.exec_tag) if x is not None else None
    self.active_se = x.se if x is not None else None
    self.active_blob = (ctypes.c_ubyte * len(x.blob)).from_buffer_copy(x.blob) if x is not None else None
    return self.active_blob

  def on_occupancy_ev(self, ev:rocprof.rocprofiler_thread_trace_decoder_occupancy_t):
    if DEBUG >= 5: print(f"OCC {ev.time=} {self.active_se=} {ev.cu=} {ev.simd=} {ev.wave_id=} {ev.start=}")
    self.occ_events.setdefault(unwrap(self.active_run), []).append(OccEvent(ev.wave_id, ev.cu, ev.simd, unwrap(self.active_se), ev.time, ev.start))

  def on_wave_ev(self, ev:rocprof.rocprofiler_thread_trace_decoder_wave_t):
    if DEBUG >= 5: print(f"WAVE {ev.wave_id=} {self.active_se=} {ev.cu=} {ev.simd=} {ev.contexts=} {ev.begin_time=} {ev.end_time=}")
    # Skip wave events without instruction timings, occupancy events give the start and duration.
    if ev.instructions_size == 0: return

    insts_blob = bytearray(sz:=ev.instructions_size * ctypes.sizeof(rocprof.rocprofiler_thread_trace_decoder_inst_t))
    ctypes.memmove((ctypes.c_char * sz).from_buffer(insts_blob), ev.instructions_array, sz)

    self.inst_execs.setdefault(unwrap(self.active_run), []).append(WaveExec(ev.wave_id, ev.cu, ev.simd, unwrap(self.active_se), ev.begin_time,
                                                                             ev.end_time, insts_blob))

def decode(sqtt_evs:list[ProfileSQTTEvent], disasms:dict[str, dict[int, tuple[str, int]]]) -> _ROCParseCtx:
  ROCParseCtx = _ROCParseCtx(sqtt_evs, disasms)

  @rocprof.rocprof_trace_decoder_se_data_callback_t
  def copy_cb(buf, buf_size, _):
    if (prof_info:=ROCParseCtx.next_sqtt()) is None: return 0
    buf[0] = ctypes.cast(prof_info, ctypes.POINTER(ctypes.c_ubyte))
    buf_size[0] = len(prof_info)
    return len(prof_info)

  @rocprof.rocprof_trace_decoder_trace_callback_t
  def trace_cb(record_type, events_ptr, n, _):
    match record_type:
      case rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
        for ev in (rocprof.rocprofiler_thread_trace_decoder_occupancy_t * n).from_address(events_ptr): ROCParseCtx.on_occupancy_ev(ev)
      case rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
        for ev in (rocprof.rocprofiler_thread_trace_decoder_wave_t * n).from_address(events_ptr): ROCParseCtx.on_wave_ev(ev)
      case rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME:
        if DEBUG >= 5:
          pairs = [(ev.shader_clock, ev.realtime_clock) for ev in (rocprof.rocprofiler_thread_trace_decoder_realtime_t * n).from_address(events_ptr)]
          print(f"REALTIME {pairs}")
      case _:
        if DEBUG >= 5: print(rocprof.enum_rocprofiler_thread_trace_decoder_record_type_t.get(record_type), events_ptr, n)
    return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

  @rocprof.rocprof_trace_decoder_isa_callback_t
  def isa_cb(instr_ptr, mem_size_ptr, size_ptr, pc, _):
    instr, mem_size_ptr[0] = ROCParseCtx.disasms[unwrap(ROCParseCtx.active_run)[0]][pc.address]

    # this is the number of bytes to next instruction, set to 0 for end_pgm
    if instr == "s_endpgm": mem_size_ptr[0] = 0
    if (max_sz:=size_ptr[0]) == 0: return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES

    # truncate the instr if it doesn't fit
    if (str_sz:=len(instr_bytes:=instr.encode()))+1 > max_sz: str_sz = max_sz
    ctypes.memmove(instr_ptr, instr_bytes, str_sz)
    size_ptr[0] = str_sz

    return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

  def worker():
    try: rocprof.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, None)
    except AttributeError as e: raise RuntimeError("Failed to find rocprof-trace-decoder. Run sudo ./extra/sqtt/install_sqtt_decoder.py to install") from e
  (t:=threading.Thread(target=worker, daemon=True)).start()
  t.join()
  return ROCParseCtx

def print_pmc(events:list[ProfilePMCEvent]) -> None:
  from tinygrad.viz.serve import unpack_pmc
  from tabulate import tabulate
  for e in events:
    print("**", e.kern)
    data = unpack_pmc(e)
    print(tabulate([r[:-1] for r in data["rows"]], headers=data["cols"], tablefmt="github"))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', type=pathlib.Path, help='Path to profile', default=pathlib.Path(temp("profile.pkl", append_user=True)))
  args = parser.parse_args()

  with args.profile.open("rb") as f: profile = pickle.load(f)
  #rctx = decode(profile, disasm)
  #print('SQTT:', rctx.inst_execs.keys())

  print_pmc([ev for ev in profile if isinstance(ev, ProfilePMCEvent)])
