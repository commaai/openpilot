#!/usr/bin/env python3
import ctypes, pathlib, argparse, pickle, dataclasses, threading, itertools
from decimal import Decimal
from typing import Generator
from tinygrad.helpers import temp, unwrap, DEBUG
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from tinygrad.runtime.autogen import rocprof
from tinygrad.renderer.amd.dsl import Inst
from tinygrad.helpers import ProfileEvent, ProfileRangeEvent, ProfilePointEvent
from tinygrad.device import ProfileProgramEvent
from test.amd.disasm import disasm

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
  def wave_loc(self) -> str: return f"{self.cu_loc} SIMD:{self.simd} W:{self.wave_id}"

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
  def __init__(self, sqtt_evs:list[ProfileSQTTEvent], disasms:dict[str, dict[int, Inst]]):
    self.sqtt_evs, self.disasms = iter(sqtt_evs), {k:{k2:(disasm(v2), v2.size()) for k2,v2 in v.items()} for k,v in disasms.items()}
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

def decode(sqtt_evs:list[ProfileSQTTEvent], disasms:dict[str, dict[int, Inst]]) -> _ROCParseCtx:
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

  exc:Exception|None = None
  def worker():
    nonlocal exc
    try: rocprof.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, None)
    except AttributeError as e:
      exc = RuntimeError("Failed to find rocprof-trace-decoder. Run sudo ./extra/sqtt/install_rocprof_decoder.py to install")
      exc.__cause__ = e
  (t:=threading.Thread(target=worker, daemon=True)).start()
  t.join()
  if exc is not None:
    raise exc
  return ROCParseCtx

def unpack_occ(viz_data, i:int, j:int, key:tuple[str, int], data:list, p:ProfileProgramEvent, target:str) -> dict:
  from tinygrad.viz.serve import amd_decode, create_step, row_tuple
  steps = viz_data.ctxs[i]["steps"]
  if len(steps[j+1:]) > 0: return {"steps":[{k:v for k,v in s.items() if k != "data"} for s in steps[j+1:]]}
  base = unwrap(p.base)
  disasm:dict[int, Inst] = {addr+base:inst for addr,inst in amd_decode(unwrap(p.lib), target).items()}
  rctx = decode(data, {p.tag:disasm})
  cu_events:dict[str, list[ProfileEvent]] = {}
  # ** inst traces
  wave_insts:dict[str, dict[str, dict]] = {}
  inst_units:dict[str, itertools.count] = {}
  for w in rctx.inst_execs.get(key, []):
    if (u:=w.wave_loc) not in inst_units: inst_units[u] = itertools.count(0)
    n = next(inst_units[u])
    if (events:=cu_events.get(w.cu_loc)) is None: cu_events[w.cu_loc] = events = []
    events.append(ProfileRangeEvent(f"SIMD:{w.simd}", loc:=f"INST WAVE:{w.wave_id} N:{n}", Decimal(w.begin_time), Decimal(w.end_time)))
    wave_insts.setdefault(w.cu_loc, {})[f"{u} N:{n}"] = {"wave":w, "disasm":disasm, "prg":p, "run_number":n, "loc":loc}
  # ** occ traces (only WAVESTART/WAVEEND)
  units:dict[str, itertools.count] = {}
  wave_start:dict[str, int] = {}
  for occ in rctx.occ_events.get(key, []):
    if (u:=occ.wave_loc) not in units: units[u] = itertools.count(0)
    if u in inst_units: continue
    if occ.start: wave_start[u] = occ.time
    else:
      if (events:=cu_events.get(occ.cu_loc)) is None: cu_events[occ.cu_loc] = events = []
      events.append(ProfileRangeEvent(f"SIMD:{occ.simd}", f"OCC WAVE:{occ.wave_id} N:{next(units[u])}", Decimal(wave_start.pop(u)),Decimal(occ.time)))
  # ** split graph by CU
  for cu in sorted(cu_events, key=row_tuple):
    steps.append(create_step(f"{cu} {len(cu_events[cu])}", ("/cu-sqtt", i, len(steps)), depth=1,
                             data=[ProfilePointEvent(unit, "start", unit, ts=Decimal(0)) for unit in units]+cu_events[cu]))
    for k in sorted(wave_insts.get(cu, []), key=row_tuple):
      wd = wave_insts[cu][k]
      steps.append(create_step(k.replace(cu, ""), ("/amd-sqtt-insts", i, len(steps)), loc=wd["loc"], depth=2,
                               data={"fxn":unpack_insts, "args":(wd,)}))
  return {"steps":[{k:v for k,v in s.items() if k != "data"} for s in steps[j+1:]]}

def unpack_insts(viz_data, i:int, j:int, data:dict) -> dict:
  columns = ["PC", "Instruction", "Hits", "Cycles", "Stall", "Type"]
  inst_columns = ["N", "Clk", "Idle", "Dur", "Stall"]
  # Idle:     The total time gap between the completion of previous instruction and the beginning of the current instruction.
  #           The idle time can be caused by:
  #             * Arbiter loss
  #             * Source or destination register dependency
  #             * Instruction cache miss
  # Stall:    The total number of cycles the hardware pipe couldn't issue an instruction.
  # Duration: Total latency in cycles, defined as "Stall time + Issue time" for gfx9 or "Stall time + Execute time" for gfx10+.
  prev_instr = (w:=data["wave"]).begin_time
  pc_to_inst = data["disasm"]
  start_pc = None
  rows:dict[int, dict] = {}
  for pc, inst in pc_to_inst.items():
    if start_pc is None: start_pc = pc
    rows[pc] = {"pc":pc-start_pc, "inst":str(inst), "hit_count":0, "dur":0, "stall":0, "type":"", "hits":{"cols":inst_columns, "rows":[]}}
  for e in w.unpack_insts():
    if not (inst:=rows[e.pc]).get("type"): inst["type"] = str(e.typ).split("_")[-1]
    inst["hit_count"] += 1
    inst["dur"] += e.dur
    inst["stall"] += e.stall
    inst["hits"]["rows"].append((inst["hit_count"]-1, e.time, max(0, e.time-prev_instr), e.dur, e.stall))
    prev_instr = max(prev_instr, e.time + e.dur)
  summary = [{"label":"Total Cycles", "value":w.end_time-w.begin_time}, {"label":"SE", "value":w.se}, {"label":"CU", "value":w.cu},
             {"label":"SIMD", "value":w.simd}, {"label":"Wave ID", "value":w.wave_id}, {"label":"Run number", "value":data["run_number"]}]
  return {"rows":[tuple(v.values()) for v in rows.values()], "cols":columns, "metadata":[summary], "ref":viz_data.ref_map.get(data["prg"].name)}

def print_data(data:dict) -> None:
  from tabulate import tabulate
  # plaintext
  if "src" in data: print(data["src"])
  # table format
  elif "cols" in data:
    print(tabulate([r[:len(data["cols"])] for r in data["rows"]], headers=data["cols"], tablefmt="github"))

def main() -> None:
  import tinygrad.viz.serve as viz
  from tinygrad.uop.ops import RewriteTrace
  data = viz.VizData()

  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', type=pathlib.Path, metavar="PATH", help='Path to profile (optional file, default: latest profile)',
                      default=pathlib.Path(temp("profile.pkl", append_user=True)))
  parser.add_argument('--kernel', type=str, default=None, metavar="NAME", help='Kernel to focus on (optional name, default: all kernels)')
  parser.add_argument('-n', type=int, default=3, metavar="NUM", help='Max traces to print (optional number, default: 3 traces)')
  args = parser.parse_args()

  with args.profile.open("rb") as f: profile = pickle.load(f)

  viz.get_profile(profile, data=data)

  # List all kernels
  if args.kernel is None:
    for c in data.ctxs:
      print(c["name"])
      for s in c["steps"]: print("  "+s["name"])
    return None

  # Find kernel trace
  trace = next((c for c in data.ctxs if c["name"] == f"SQTT {args.kernel}"), None)
  if not trace: raise RuntimeError(f"no matching trace for {args.kernel}")
  n = 0
  for s in trace["steps"]:
    if "PKTS" in s["name"]: continue
    print(s["name"])
    ret = viz.get_render(data, s["query"])
    print_data(ret)
    n += 1
    if n > args.n: break

if __name__ == "__main__":
  main()
