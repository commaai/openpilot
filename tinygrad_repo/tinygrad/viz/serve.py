#!/usr/bin/env python3
import multiprocessing, pickle, difflib, os, threading, json, time, sys, webbrowser, socket, argparse, functools, codecs, io, struct
import pathlib, traceback, itertools, socketserver
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from decimal import Decimal
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler
from typing import Any, TypedDict, TypeVar, Generator, Callable
from tinygrad.helpers import colored, getenv, tqdm, unwrap, word_wrap, TRACEMETA, ProfileEvent, ProfileRangeEvent, TracingKey, ProfilePointEvent, temp
from tinygrad.helpers import printable

# NOTE: using HTTPServer forces a potentially slow socket.getfqdn
class TCPServerWithReuse(socketserver.TCPServer):
  allow_reuse_address = True
  def __init__(self, server_address, RequestHandlerClass):
    print(f"*** started server on http://127.0.0.1:{server_address[1]}")
    super().__init__(server_address, RequestHandlerClass)

class HTTPRequestHandler(BaseHTTPRequestHandler):
  def send_data(self, data:bytes, content_type:str="application/json", status_code:int=200):
    self.send_response(status_code)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    return self.wfile.write(data)
  def stream_json(self, source:Generator):
    try:
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-cache")
      self.end_headers()
      for r in source:
        self.wfile.write(f"data: {json.dumps(r)}\n\n".encode("utf-8"))
        self.wfile.flush()
      self.wfile.write("data: [DONE]\n\n".encode("utf-8"))
    # pass if client closed connection
    except (BrokenPipeError, ConnectionResetError): return

from tinygrad.uop.ops import TrackedGraphRewrite, RewriteTrace, UOp, Ops, GroupOp, srender, sint, sym_infer, range_str, pyrender
from tinygrad.uop.ops import print_uops, range_start, multirange_str
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent, ProfileGraphEntry, Device, ProfileProgramEvent
from tinygrad.renderer import ProgramSpec
from tinygrad.dtype import dtypes

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.VCONST: "#e0e0e0", Ops.REDUCE: "#FF5B5B",
               Ops.PARAM:"#cb9037", **{x:"#f2cb91" for x in {Ops.DEFINE_LOCAL, Ops.DEFINE_REG}}, Ops.REDUCE_AXIS: "#FF6B6B",
               Ops.RANGE: "#c8a0e0", Ops.ASSIGN: "#909090", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#cef263", Ops.WMMA: "#efefc0", Ops.MULTI: "#f6ccff", Ops.KERNEL: "#3e7f55", Ops.CUSTOM_KERNEL: "#3ebf55",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80",
               Ops.BUFFER_VIEW: "#E5EAFF", Ops.BUFFER: "#B0BDFF", Ops.COPY: "#a040a0", Ops.ENCDEC: "#bf71b6",
               Ops.CALL: "#00B7C8", Ops.PARAM: "#14686F",
               Ops.ALLREDUCE: "#ff40a0", Ops.MSELECT: "#d040a0", Ops.MSTACK: "#d040a0", Ops.CONTIGUOUS: "#FFC14D",
               Ops.BUFFERIZE: "#FF991C", Ops.REWRITE_ERROR: "#ff2e2e", Ops.AFTER: "#8A7866", Ops.END: "#524C46"}

# VIZ API


# A step is a lightweight descriptor for a trace entry
# Includes a name, metadata and a URL path for fetching the full data

def create_step(name:str, query:tuple[str, int, int], data=None, depth:int=0, **kwargs) -> dict:
  return {"name":name, "query":f"{query[0]}?ctx={query[1]}&step={query[2]}", "data":data, "depth":depth, **kwargs}

# ** list all saved rewrites

ref_map:dict[Any, int] = {}
def get_rewrites(t:RewriteTrace) -> list[dict]:
  ret = []
  for i,(k,v) in enumerate(zip(t.keys, t.rewrites)):
    steps = [create_step(s.name, ("/graph-rewrites", i, j), loc=s.loc, match_count=len(s.matches), code_line=printable(s.loc),
                         trace=k.tb if j==0 else None, depth=s.depth) for j,s in enumerate(v)]
    if isinstance(k.ret, ProgramSpec):
      steps.append(create_step("View UOp List", ("/uops", i, len(steps)), k.ret))
      steps.append(create_step("View Program", ("/code", i, len(steps)), k.ret))
      steps.append(create_step("View Disassembly", ("/asm", i, len(steps)), k.ret))
    for key in k.keys: ref_map[key] = i
    ret.append({"name":k.display_name, "steps":steps})
  return ret

# ** get the complete UOp graphs for one rewrite

class GraphRewriteDetails(TypedDict):
  graph: dict                            # JSON serialized UOp for this rewrite step
  uop: str                               # strigified UOp for this rewrite step
  diff: list[str]|None                   # diff of the single UOp that changed
  change: list[int]|None                 # the new UOp id + all its parents ids
  upat: tuple[tuple[str, int], str]|None # [loc, source_code] of the matched UPat

def shape_to_str(s:tuple[sint, ...]): return "(" + ','.join(srender(x) for x in s) + ")"
def mask_to_str(s:tuple[tuple[sint, sint], ...]): return "(" + ','.join(shape_to_str(x) for x in s) + ")"
def pystr(u:UOp) -> str:
   # pyrender may check for shape mismatch
  try: return pyrender(u)
  except Exception: return str(u)

def uop_to_json(x:UOp) -> dict[int, dict]:
  assert isinstance(x, UOp)
  graph: dict[int, dict] = {}
  excluded: set[UOp] = set()
  for u in (toposort:=x.toposort()):
    # always exclude DEVICE/CONST/UNIQUE
    if u.op in {Ops.DEVICE, Ops.CONST, Ops.UNIQUE, Ops.LUNIQUE} and u is not x: excluded.add(u)
    if u.op is Ops.VCONST and u.dtype.scalar() == dtypes.index and u is not x: excluded.add(u)
    if u.op is Ops.VECTORIZE and len(u.src) == 0: excluded.add(u)
  for u in toposort:
    if u in excluded: continue
    argst = codecs.decode(str(u.arg), "unicode_escape")
    if u.op in GroupOp.Movement: argst = (mask_to_str if u.op in {Ops.SHRINK, Ops.PAD} else shape_to_str)(u.marg)
    if u.op is Ops.KERNEL:
      ast_str = f"SINK{tuple(s.op for s in u.arg.ast.src)}" if u.arg.ast.op is Ops.SINK else repr(u.arg.ast.op)
      argst = f"<Kernel {len(list(u.arg.ast.toposort()))} {ast_str} {[str(m) for m in u.arg.metadata]}>"
    label = f"{str(u.op).split('.')[1]}{(chr(10)+word_wrap(argst.replace(':', ''))) if u.arg is not None else ''}"
    if u.dtype != dtypes.void: label += f"\n{u.dtype}"
    for idx,x in enumerate(u.src[:1] if u.op in {Ops.BUFFERIZE, Ops.INDEX} else (u.src if u.op is not Ops.END else [])):
      if x in excluded:
        arg = f"{x.arg:g}" if x.op is Ops.CONST and dtypes.is_float(x.dtype) else f"{x.arg}"
        label += f"\n{x.op.name}{idx} {arg}" + (f" {x.src[0].op}" if len(x.src) else "")
    try:
      if len(rngs:=u.ranges):
        label += f"\n({multirange_str(rngs, color=True)})"
      if u._shape is not None:
        label += f"\n{shape_to_str(u.shape)}"
      if u.op in {Ops.INDEX, Ops.BUFFERIZE}:
        if len(u.toposort()) < 30: label += f"\n{u.render()}"
        ranges: list[UOp] = []
        for us in u.src[1:]: ranges += [s for s in us.toposort() if s.op in {Ops.RANGE, Ops.SPECIAL}]
        if ranges: label += "\n"+' '.join([f"{s.render()}={s.vmax+1}" for s in ranges])
      if u.op in {Ops.END, Ops.REDUCE} and len(trngs:=list(UOp.sink(*u.src[range_start[u.op]:]).ranges)):
        label += "\n"+' '.join([f"{range_str(s, color=True)}({s.vmax+1})" for s in trngs])
    except Exception:
      label += "\n<ISSUE GETTING LABEL>"
    if (ref:=ref_map.get(u.arg.ast) if u.op is Ops.KERNEL else None) is not None: label += f"\ncodegen@{ctxs[ref]['name']}"
    # NOTE: kernel already has metadata in arg
    if TRACEMETA >= 2 and u.metadata is not None and u.op is not Ops.KERNEL: label += "\n"+str(u.metadata)
    graph[id(u)] = {"label":label, "src":[(i,id(x)) for i,x in enumerate(u.src) if x not in excluded], "color":uops_colors.get(u.op, "#ffffff"),
                    "ref":ref, "tag":repr(u.tag) if u.tag is not None else None}
  return graph

@functools.cache
def _reconstruct(a:int):
  op, dtype, src, arg, *rest = trace.uop_fields[a]
  arg = type(arg)(_reconstruct(arg.ast), arg.metadata) if op is Ops.KERNEL else arg
  return UOp(op, dtype, tuple(_reconstruct(s) for s in src), arg, *rest)

def get_full_rewrite(ctx:TrackedGraphRewrite) -> Generator[GraphRewriteDetails, None, None]:
  next_sink = _reconstruct(ctx.sink)
  # in the schedule graph we don't show indexing ops (unless it's in a kernel AST or rewriting dtypes.index sink)
  yield {"graph":uop_to_json(next_sink), "uop":pystr(next_sink), "change":None, "diff":None, "upat":None}
  replaces: dict[UOp, UOp] = {}
  for u0_num,u1_num,upat_loc,dur in tqdm(ctx.matches):
    replaces[u0:=_reconstruct(u0_num)] = u1 = _reconstruct(u1_num)
    try: new_sink = next_sink.substitute(replaces)
    except RuntimeError as e: new_sink = UOp(Ops.NOOP, arg=str(e))
    match_repr = f"# {dur*1e6:.2f} us\n"+printable(upat_loc)
    yield {"graph":(sink_json:=uop_to_json(new_sink)), "uop":pystr(new_sink), "change":[id(x) for x in u1.toposort() if id(x) in sink_json],
           "diff":list(difflib.unified_diff(pystr(u0).splitlines(), pystr(u1).splitlines())), "upat":(upat_loc, match_repr)}
    if not ctx.bottom_up: next_sink = new_sink

# encoder helpers

def enum_str(s, cache:dict[str, int]) -> int:
  if (cret:=cache.get(s)) is not None: return cret
  cache[s] = ret = len(cache)
  return ret

def option(s:int|None) -> int: return 0 if s is None else s+1

# Profiler API

device_ts_diffs:dict[str, tuple[Decimal, Decimal]] = {}
def cpu_ts_diff(device:str, thread=0) -> Decimal: return device_ts_diffs.get(device, (Decimal(0),))[thread]

device_props:dict[str, dict] = {}

DevEvent = ProfileRangeEvent|ProfileGraphEntry|ProfilePointEvent
def flatten_events(profile:list[ProfileEvent]) -> Generator[tuple[Decimal, Decimal, DevEvent], None, None]:
  for e in profile:
    if isinstance(e, ProfileRangeEvent): yield (e.st+(diff:=cpu_ts_diff(e.device, e.is_copy)), (e.en if e.en is not None else e.st)+diff, e)
    elif isinstance(e, ProfilePointEvent): yield (e.ts, e.ts, e)
    elif isinstance(e, ProfileGraphEvent):
      cpu_ts = []
      for ent in e.ents: cpu_ts += [e.sigs[ent.st_id]+(diff:=cpu_ts_diff(ent.device, ent.is_copy)), e.sigs[ent.en_id]+diff]
      yield (st:=min(cpu_ts)), (et:=max(cpu_ts)), ProfileRangeEvent(f"{e.ents[0].device.split(':')[0]} Graph", f"batched {len(e.ents)}", st, et)
      for i,ent in enumerate(e.ents): yield (cpu_ts[i*2], cpu_ts[i*2+1], ent)

# normalize event timestamps and attach kernel metadata
def timeline_layout(dev_events:list[tuple[int, int, float, DevEvent]], start_ts:int, scache:dict[str, int]) -> bytes|None:
  events:list[bytes] = []
  exec_points:dict[str, ProfilePointEvent] = {}
  for st,et,dur,e in dev_events:
    if isinstance(e, ProfilePointEvent) and e.name == "exec": exec_points[e.arg["name"]] = e
    if dur == 0: continue
    name, fmt, key = e.name, [], None
    if (ref:=ref_map.get(name)) is not None:
      name = ctxs[ref]["name"]
      if isinstance(p:=trace.keys[ref].ret, ProgramSpec) and (ei:=exec_points.get(p.name)) is not None:
        flops = sym_infer(p.estimates.ops, var_vals:=ei.arg['var_vals'])/(t:=dur*1e-6)
        membw, ldsbw = sym_infer(p.estimates.mem, var_vals)/t, sym_infer(p.estimates.lds, var_vals)/t
        fmt = [f"{flops*1e-9:.0f} GFLOPS" if flops < 1e14 else f"{flops*1e-12:.0f} TFLOPS",
              (f"{membw*1e-9:.0f} GB/s" if membw < 1e13 else f"{membw*1e-12:.0f} TB/s")+" mem",
              (f"{ldsbw*1e-9:.0f} GB/s" if ldsbw < 1e15 else f"{ldsbw*1e-12:.0f} TB/s")+" lds"]
        if (metadata_str:=",".join([str(m) for m in (ei.arg['metadata'] or ())])): fmt.append(metadata_str)
        if isinstance(e, ProfileGraphEntry): fmt.append("(batched)")
        key = ei.key
    elif isinstance(e.name, TracingKey):
      name = e.name.display_name
      ref = next((v for k in e.name.keys if (v:=ref_map.get(k)) is not None), None)
      if isinstance(e.name.ret, str): fmt.append(e.name.ret)
    events.append(struct.pack("<IIIIfI", enum_str(name, scache), option(ref), option(key), st-start_ts, dur, enum_str("\n".join(fmt), scache)))
  return struct.pack("<BI", 0, len(events))+b"".join(events) if events else None

def encode_mem_free(key:int, ts:int, execs:list[ProfilePointEvent], scache:dict) -> bytes:
  ei_encoding:list[tuple[int, int, int, int]] = [] # <[u32, u32, u32, u8] [run id, display name, buffer number and mode (2 = r/w, 1 = w, 0 = r)]
  for e in execs:
    num = next(i for i,k in enumerate(e.arg["bufs"]) if k == key)
    mode = 2 if (num in e.arg["inputs"] and num in e.arg["outputs"]) else 1 if (num in e.arg["outputs"]) else 0
    ei_encoding.append((e.key, enum_str(e.arg["name"], scache), num, mode))
  return struct.pack("<BIII", 0, ts, key, len(ei_encoding))+b"".join(struct.pack("<IIIB", *t) for t in ei_encoding)

def mem_layout(dev_events:list[tuple[int, int, float, DevEvent]], start_ts:int, end_ts:int, peaks:list[int], dtype_size:dict[str, int],
               scache:dict[str, int]) -> bytes|None:
  peak, mem = 0, 0
  temp:dict[int, int] = {}
  events:list[bytes] = []
  buf_ei:dict[int, list[ProfilePointEvent]] = {}

  for st,_,_,e in dev_events:
    if not isinstance(e, ProfilePointEvent): continue
    if e.name == "alloc":
      safe_sz = min(1_000_000_000_000, e.arg["sz"])
      events.append(struct.pack("<BIIIQ", 1, int(e.ts)-start_ts, e.key, enum_str(e.arg["dtype"].name, scache), safe_sz))
      dtype_size.setdefault(e.arg["dtype"].name, e.arg["dtype"].itemsize)
      temp[e.key] = nbytes = safe_sz*e.arg["dtype"].itemsize
      mem += nbytes
      if mem > peak: peak = mem
    if e.name == "exec" and e.arg["bufs"]:
      for b in e.arg["bufs"]: buf_ei.setdefault(b, []).append(e)
    if e.name == "free":
      events.append(encode_mem_free(e.key, int(e.ts) - start_ts, buf_ei.pop(e.key, []), scache))
      mem -= temp.pop(e.key)
  for t in temp: events.append(encode_mem_free(t, end_ts-start_ts, buf_ei.pop(t, []), scache))
  peaks.append(peak)
  return struct.pack("<BIQ", 1, len(events), peak)+b"".join(events) if events else None

# by default, VIZ does not start when there is an error
# use this to instead display the traceback to the user
@contextmanager
def soft_err(fn:Callable):
  try: yield
  except Exception: fn({"src":traceback.format_exc()})

def row_tuple(row:str) -> tuple[tuple[int, int], ...]:
  return tuple((ord(ss[0][0]), int(ss[1])) if len(ss:=x.split(":"))>1 else (999,999) for x in row.split())

# *** Performance counters

metrics:dict[str, Callable[[dict[str, tuple[int, int, int]]], str]] = {
  "VALU utilization": lambda s: f"{100 * (s['SQ_INSTS_VALU'][0] / s['SQ_INSTS_VALU'][2]) / (s['GRBM_GUI_ACTIVE'][1] * 4):.1f}%",
  "SALU utilization": lambda s: f"{100 * (s['SQ_INSTS_SALU'][0] / s['SQ_INSTS_SALU'][2]) / (s['GRBM_GUI_ACTIVE'][1] * 4):.1f}%",
}

def unpack_pmc(e) -> dict:
  agg_cols = ["Name", "Sum"]
  sample_cols = ["XCC", "INST", "SE", "SA", "WGP", "Value"]
  rows:list[list] = []
  stats:dict[str, tuple[int, int, int]] = {}  # name -> (sum, max, count)
  view, ptr = memoryview(e.blob).cast('Q'), 0
  for s in e.sched:
    row:list = [s.name, 0, {"cols":sample_cols, "rows":[]}]
    max_val, cnt = 0, 0
    for sample in itertools.product(range(s.xcc), range(s.inst), range(s.se), range(s.sa), range(s.wgp)):
      row[1] += (val:=int(view[ptr]))
      max_val, cnt = max(max_val, val), cnt + 1
      row[2]["rows"].append(sample+(val,))
      ptr += 1
    stats[s.name] = (row[1], max_val, cnt)
    rows.append(row)
  for name, fn in metrics.items():
    try: rows.append([name, fn(stats)])
    except KeyError: pass
  return {"rows":rows, "cols":agg_cols}

# ** on startup, list all the performance counter traces

def load_counters(profile:list[ProfileEvent]) -> None:
  from tinygrad.runtime.ops_amd import ProfileSQTTEvent, ProfilePMCEvent
  counter_events:dict[tuple[int, int], dict] = {}
  durations:dict[str, list[float]] = {}
  prg_events:dict[int, ProfileProgramEvent] = {}
  for e in profile:
    if isinstance(e, (ProfilePMCEvent, ProfileSQTTEvent)): counter_events.setdefault((e.kern, e.exec_tag), {}).setdefault(type(e), []).append(e)
    if isinstance(e, ProfileRangeEvent) and e.device.startswith("AMD") and e.en is not None:
      durations.setdefault(str(e.name), []).append(float(e.en-e.st))
    if isinstance(e, ProfileProgramEvent) and e.tag is not None: prg_events[e.tag] = e
  if len(counter_events) == 0: return None
  ctxs.append({"name":"All Counters", "steps":[create_step("PMC", ("/all-pmc", len(ctxs), 0), (durations, all_counters:={}))]})
  run_number = {n:0 for n,_ in counter_events}
  for (k, tag),v in counter_events.items():
    # use the colored name if it exists
    name = trace.keys[r].ret.name if (r:=ref_map.get(pname:=prg_events[k].name)) is not None else pname
    run_number[k] += 1
    steps:list[dict] = []
    if (pmc:=v.get(ProfilePMCEvent)):
      steps.append(create_step("PMC", ("/prg-pmc", len(ctxs), len(steps)), pmc))
      all_counters[(name, run_number[k], pname)] = pmc[0]
    # to decode a SQTT trace, we need the raw stream, program binary and device properties
    if (sqtt:=v.get(ProfileSQTTEvent)):
      for e in sqtt:
        if e.itrace: steps.append(create_step(f"PKTS SE:{e.se}", (f"/prg-pkts-{e.se}", len(ctxs), len(steps)),
                                              data=(e.blob, prg_events[k].lib, device_props[e.device]["gfx_target_version"])))
      steps.append(create_step("SQTT", ("/prg-sqtt", len(ctxs), len(steps)), ((k, tag), sqtt, prg_events[k])))
    ctxs.append({"name":f"Exec {name}"+(f" n{run_number[k]}" if run_number[k] > 1 else ""), "steps":steps})

def sqtt_timeline(data:bytes, lib:bytes, target:int) -> list[ProfileEvent]:
  from extra.assembly.amd.sqttmap import map_insts, InstructionInfo
  from extra.assembly.amd.sqtt import PacketType, INST, InstOp, VALUINST, IMMEDIATE, IMMEDIATE_MASK, VMEMEXEC, ALUEXEC
  ret:list[ProfileEvent] = []
  rows:dict[str, None] = {}
  trace:dict[str, set[int]] = {}
  def add(name:str, p:PacketType, idx=0, width=1, op_name=None, wave=None, info:InstructionInfo|None=None) -> None:
    if hasattr(p, "wave"): wave = p.wave
    rows.setdefault(r:=(f"WAVE:{wave}" if wave is not None else f"{p.__class__.__name__}:0 {name}"))
    key = TracingKey(f"{op_name if op_name is not None else name} OP:{idx}", ret=info.inst.disasm() if info is not None else None)
    ret.append(ProfileRangeEvent(r, key, Decimal(p._time), Decimal(p._time+width)))
  for p, info in map_insts(data, lib, target):
    if len(ret) > getenv("MAX_SQTT_PKTS", 50_000): break
    if isinstance(p, INST):
      op_name = p.op.name if isinstance(p.op, InstOp) else f"0x{p.op:02x}"
      name, width = (op_name, 10 if "BARRIER" in op_name else 1)
      add(name, p, width=width, idx=int("OTHER" in name), info=info)
    if isinstance(p, (VALUINST, IMMEDIATE)): add(p.__class__.__name__, p, info=info)
    if isinstance(p, IMMEDIATE_MASK): add("IMMEDIATE", p, wave=unwrap(info.wave), info=info)
    if isinstance(p, (VMEMEXEC, ALUEXEC)):
      name = str(p.src).split('.')[1]
      if name == "VALU_SALU":
        add("VALU", p)
        add("SALU", p)
      else:
        add(name.replace("_ALT", ""), p, op_name=name)
      if p._time in trace.setdefault(name, set()): raise AssertionError(f"packets overlap in shared resource! {name}")
      trace[name].add(p._time)
  return [ProfilePointEvent(r, "start", r, ts=Decimal(0)) for r in rows]+ret

# ** SQTT OCC only unpacks wave start, end time and SIMD location

def unpack_sqtt(key:tuple[str, int], data:list, p:ProfileProgramEvent) -> tuple[dict[str, list[ProfileEvent]], list[str], dict[str, dict[str, dict]]]:
  # * init decoder
  from extra.sqtt.roc import decode
  base = unwrap(p.base)
  addr_table = amd_decode(unwrap(p.lib), device_props[p.device]["gfx_target_version"])
  disasm:dict[int, tuple[str, int]] = {addr+base:(inst.disasm(), inst.size()) for addr, inst in addr_table.items()}
  rctx = decode(data, {p.tag:disasm})
  cu_events:dict[str, list[ProfileEvent]] = {}
  # * INST waves
  wave_insts:dict[str, dict[str, dict]] = {}
  inst_units:dict[str, itertools.count] = {}
  for w in rctx.inst_execs.get(key, []):
    if (u:=w.wave_loc) not in inst_units: inst_units[u] = itertools.count(0)
    n = next(inst_units[u])
    if (events:=cu_events.get(w.cu_loc)) is None: cu_events[w.cu_loc] = events = []
    events.append(ProfileRangeEvent(f"SIMD:{w.simd}", loc:=f"INST WAVE:{w.wave_id} N:{n}", Decimal(w.begin_time), Decimal(w.end_time)))
    wave_insts.setdefault(w.cu_loc, {})[f"{u} N:{n}"] = {"wave":w, "disasm":disasm, "prg":p, "run_number":n, "loc":loc}
  # * OCC waves
  units:dict[str, itertools.count] = {}
  wave_start:dict[str, int] = {}
  for occ in rctx.occ_events.get(key, []):
    if (u:=occ.wave_loc) not in units: units[u] = itertools.count(0)
    if u in inst_units: continue
    if occ.start: wave_start[u] = occ.time
    else:
      if (events:=cu_events.get(occ.cu_loc)) is None: cu_events[occ.cu_loc] = events = []
      events.append(ProfileRangeEvent(f"SIMD:{occ.simd}", f"OCC WAVE:{occ.wave_id} N:{next(units[u])}", Decimal(wave_start.pop(u)),Decimal(occ.time)))
  return cu_events, list(units), wave_insts

def device_sort_fn(k:str) -> tuple[int, str, int]:
  order = {"GC": 0, "USER": 1, "TINY": 2, "DISK": 999}
  dname = k.split()[0]
  dev_rank = next((v for k,v in order.items() if dname.startswith(k)), len(order))
  return (dev_rank, dname, len(k))

def get_profile(profile:list[ProfileEvent], sort_fn:Callable[[str], Any]=device_sort_fn) -> bytes|None:
  # start by getting the time diffs
  device_decoders:dict[str, Callable[[list[ProfileEvent]], None]] = {}
  for ev in profile:
    if isinstance(ev, ProfileDeviceEvent):
      device_ts_diffs[ev.device] = (ev.comp_tdiff,ev.copy_tdiff if ev.copy_tdiff is not None else ev.comp_tdiff)
      if ev.props is not None: device_props[ev.device] = ev.props
      if (d:=ev.device.split(":")[0]) == "AMD": device_decoders[d] = load_counters
  # load device specific counters
  for fxn in device_decoders.values(): fxn(profile)
  # map events per device
  dev_events:dict[str, list[tuple[int, int, float, DevEvent]]] = {}
  markers:list[ProfilePointEvent] = []
  start_ts:int|None = None
  end_ts:int|None = None
  for ts,en,e in flatten_events(profile):
    dev_events.setdefault(e.device,[]).append((st:=int(ts), et:=int(en), float(en-ts), e))
    if start_ts is None or st < start_ts: start_ts = st
    if end_ts is None or et > end_ts: end_ts = et
    if isinstance(e, ProfilePointEvent) and e.name == "marker": markers.append(e)
  if start_ts is None: return None
  # return layout of per device events
  layout:dict[str, bytes|None] = {}
  scache:dict[str, int] = {}
  peaks:list[int] = []
  dtype_size:dict[str, int] = {}
  for k,v in dev_events.items():
    v.sort(key=lambda e:e[0])
    layout[k] = timeline_layout(v, start_ts, scache)
    layout[f"{k} Memory"] = mem_layout(v, start_ts, unwrap(end_ts), peaks, dtype_size, scache)
  sorted_layout = sorted([k for k,v in layout.items() if v is not None], key=sort_fn)
  ret = [b"".join([struct.pack("<B", len(k)), k.encode(), unwrap(layout[k])]) for k in sorted_layout]
  index = json.dumps({"strings":list(scache), "dtypeSize":dtype_size, "markers":[{"ts":int(e.ts-start_ts), **e.arg} for e in markers]}).encode()
  return struct.pack("<IQII", unwrap(end_ts)-start_ts, max(peaks,default=0), len(index), len(ret))+index+b"".join(ret)

# ** Assembly static analyzers

def get_stdout(f: Callable) -> str:
  buf = io.StringIO()
  try:
    with redirect_stdout(buf), redirect_stderr(buf): f()
  except Exception: traceback.print_exc(file=buf)
  return buf.getvalue()

def amd_readelf(lib:bytes) -> list[dict]:
  from tinygrad.runtime.autogen import amdgpu_kd
  from tinygrad.runtime.support.elf import elf_loader
  image, sections, __ = elf_loader(lib)
  rodata = next((s for s in sections if s.name == ".rodata")).content
  kd = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytearray(rodata))
  vgpr_gran = kd.compute_pgm_rsrc1 & amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT
  return [{"label":f"{resource} Alloc", "value":val} for resource,val in [("VGPR", (vgpr_gran+1)*8-7), ("LDS",kd.group_segment_fixed_size),
                                                                          ("Scratch", kd.private_segment_fixed_size)] if val > 0]

def amd_decode(lib:bytes, target:int) -> dict[int, Any]: # Any is the Inst class from extra.assembly.amd.dsl
  from tinygrad.runtime.support.elf import elf_loader
  from extra.assembly.amd import detect_format
  from extra.assembly.amd.dsl import Inst
  image, sections, _ = elf_loader(lib)
  text = next((sh for sh in sections if sh.name == ".text"), None)
  assert text is not None, "no .text section found in ELF"
  off, buf = text.header.sh_addr, text.content
  arch = {11:"rdna3", 12:"rdna4"}.get(target//10000, "cdna")
  addr_table:dict[int, Inst] = {}
  offset = 0
  while offset < len(buf):
    remaining = buf[offset:]
    fmt = detect_format(remaining, arch)
    decoded = fmt.from_bytes(remaining)
    addr_table[off+offset] = decoded
    offset += decoded.size()
  return addr_table

def parse_branch(inst) -> int|None:
  if "branch" in getattr(inst, "op_name", "").lower():
    x = inst.simm16 & 0xffff
    return (x - 0x10000 if x & 0x8000 else x)*4
  return None

COND_TAKEN, COND_NOT_TAKEN, UNCOND = range(3)
def amdgpu_cfg(lib:bytes, target:int) -> dict:
  # decode
  pc_table = amd_decode(lib, target)
  # get leaders
  leaders:set[int] = {next(iter(pc_table))}
  for pc, inst in pc_table.items():
    if (offset:=parse_branch(inst)) is not None: leaders.update((pc+inst.size()+offset, pc+inst.size()))
  # build the cfg
  curr:int|None = None
  blocks:dict[int, list[int]] = {}
  paths:dict[int, dict[int, int]] = {}
  lines:list[str] = []
  disasm = {pc:inst.disasm() for pc,inst in pc_table.items()}
  asm_width = max(len(asm) for asm in disasm.values())
  for pc, inst in pc_table.items():
    # skip instructions only used for padding
    if (asm:=disasm[pc]) == "s_code_end": continue
    lines.append(f"  {asm:<{asm_width}}  // {pc:012X}")
    if pc in leaders:
      paths[curr:=pc] = {}
      blocks[pc] = []
    else: assert curr is not None, f"no basic block found for {pc}"
    blocks[curr].append(pc)
    # otherwise a basic block can have exactly one or two paths
    nx = pc+inst.size()
    if (offset:=parse_branch(inst)) is not None:
      if inst.op_name == "S_BRANCH": paths[curr][nx+offset] = UNCOND
      else: paths[curr].update([(nx+offset, COND_TAKEN), (nx, COND_NOT_TAKEN)])
    elif nx in leaders: paths[curr][nx] = UNCOND
  pc_tokens:dict[int, list[dict]] = {}
  from extra.assembly.amd.dsl import Reg
  for pc, inst in pc_table.items():
    pc_tokens[pc] = tokens = []
    for name, field in inst._fields:
      if isinstance(val:=getattr(inst, name), Reg): tokens.append({"st":val.fmt(), "keys":[f"r{val.offset+i}" for i in range(val.sz)], "kind":1})
      elif name in {"op","opx","opy"}: tokens.append({"st":(op_name:=val.name.lower()), "keys":[op_name], "kind":0})
      elif name != "encoding" and val != field.default: tokens.append({"st":(s:=repr(val)), "keys":[s], "kind":1})
  return {"data":{"blocks":blocks, "paths":paths, "pc_tokens":pc_tokens}, "src":"\n".join(lines)}

# ** Main render function to get the complete details about a trace event

def get_render(query:str) -> dict:
  url = urlparse(query)
  i, j, fmt = get_int(qs:=parse_qs(url.query), "ctx"), get_int(qs, "step"), url.path.lstrip("/")
  data = ctxs[i]["steps"][j]["data"]
  if fmt == "graph-rewrites": return {"value":get_full_rewrite(trace.rewrites[i][j]), "content_type":"text/event-stream"}
  if fmt == "uops": return {"src":get_stdout(lambda: print_uops(data.uops or [])), "lang":"txt"}
  if fmt == "code": return {"src":data.src, "lang":"cpp"}
  if fmt == "asm":
    ret:dict = {"metadata":[]}
    if data.device.startswith("AMD") and data.lib is not None:
      with soft_err(lambda err: ret.update(err)): ret.update(amdgpu_cfg(data.lib, device_props[data.device]["gfx_target_version"]))
      with soft_err(lambda err: ret["metadata"].append(err)): ret["metadata"].append(amd_readelf(data.lib))
    else: ret["src"] = get_stdout(lambda: (compiler:=Device[data.device].compiler).disassemble(compiler.compile(data.src)))
    return ret
  if fmt == "all-pmc":
    durations, pmc = data
    ret = {"cols":{}, "rows":[]}
    for (name, n, k),events in pmc.items():
      pmc_table = unpack_pmc(events)
      ret["cols"].update([(r[0], None) for r in pmc_table["rows"]])
      ret["rows"].append((name, durations[k][n-1], *[r[1] for r in pmc_table["rows"]]))
    ret["cols"] = ["Kernel", "Duration", *ret["cols"]]
    return ret
  if fmt == "prg-pmc": return unpack_pmc(data[0])
  if fmt.startswith("prg-pkts"):
    ret = {}
    with soft_err(lambda err:ret.update(err)):
      if (events:=get_profile(sqtt_timeline(*data), sort_fn=row_tuple)): ret = {"value":events, "content_type":"application/octet-stream"}
      else: ret = {"src":"No SQTT trace on this SE."}
    return ret
  if fmt == "prg-sqtt":
    ret = {}
    if len((steps:=ctxs[i]["steps"])[j+1:]) == 0:
      with soft_err(lambda err: ret.update(err)):
        cu_events, units, wave_insts = unpack_sqtt(*data)
        for cu in sorted(cu_events, key=row_tuple):
          steps.append(create_step(f"{cu} {len(cu_events[cu])}", ("/cu-sqtt", i, len(steps)), depth=1,
                                   data=[ProfilePointEvent(unit, "start", unit, ts=Decimal(0)) for unit in units]+cu_events[cu]))
          for k in sorted(wave_insts.get(cu, []), key=row_tuple):
            steps.append(create_step(k.replace(cu, ""), ("/sqtt-insts", i, len(steps)), loc=(data:=wave_insts[cu][k])["loc"], depth=2, data=data))
    return {**ret, "steps":[{k:v for k,v in s.items() if k != "data"} for s in steps[j+1:]]}
  if fmt == "cu-sqtt": return {"value":get_profile(data, sort_fn=row_tuple), "content_type":"application/octet-stream"}
  if fmt == "sqtt-insts":
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
    for pc, (inst,_) in pc_to_inst.items():
      if start_pc is None: start_pc = pc
      rows[pc] = {"pc":pc-start_pc, "inst":inst, "hit_count":0, "dur":0, "stall":0, "type":"", "hits":{"cols":inst_columns, "rows":[]}}
    for e in w.unpack_insts():
      if not (inst:=rows[e.pc]).get("type"): inst["type"] = str(e.typ).split("_")[-1]
      inst["hit_count"] += 1
      inst["dur"] += e.dur
      inst["stall"] += e.stall
      inst["hits"]["rows"].append((inst["hit_count"]-1, e.time, max(0, e.time-prev_instr), e.dur, e.stall))
      prev_instr = max(prev_instr, e.time + e.dur)
    summary = [{"label":"Total Cycles", "value":w.end_time-w.begin_time}, {"label":"SE", "value":w.se}, {"label":"CU", "value":w.cu},
               {"label":"SIMD", "value":w.simd}, {"label":"Wave ID", "value":w.wave_id}, {"label":"Run number", "value":data["run_number"]}]
    return {"rows":[tuple(v.values()) for v in rows.values()], "cols":columns, "metadata":[summary], "ref":ref_map.get(data["prg"].name)}
  return data

# ** HTTP server

def get_int(query:dict[str, list[str]], k:str) -> int: return int(query.get(k,["0"])[0])

class Handler(HTTPRequestHandler):
  def do_GET(self):
    ret, status_code, content_type = b"", 200, "text/html"

    if (url:=urlparse(self.path)).path == "/":
      with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f: ret = f.read()
    elif self.path.startswith(("/assets/", "/js/")) and '/..' not in self.path:
      try:
        with open(os.path.join(os.path.dirname(__file__), self.path.strip('/')), "rb") as f: ret = f.read()
        if url.path.endswith(".js"): content_type = "application/javascript"
        if url.path.endswith(".css"): content_type = "text/css"
      except FileNotFoundError: status_code = 404

    elif url.path == "/ctxs":
      lst = [{**c, "steps":[{k:v for k, v in s.items() if k != "data"} for s in c["steps"]]} for c in ctxs]
      ret, content_type = json.dumps(lst).encode(), "application/json"
    elif url.path == "/get_profile" and profile_ret: ret, content_type = profile_ret, "application/octet-stream"
    else:
      if not (render_src:=get_render(self.path)): status_code = 404
      else:
        if "content_type" in render_src: ret, content_type = render_src["value"], render_src["content_type"]
        else: ret, content_type = json.dumps(render_src).encode(), "application/json"
        if content_type == "text/event-stream": return self.stream_json(render_src["value"])

    return self.send_data(ret, content_type, status_code)

# ** main loop

def reloader():
  mtime = os.stat(__file__).st_mtime
  while not stop_reloader.is_set():
    if mtime != os.stat(__file__).st_mtime:
      print("reloading server...")
      os.execv(sys.executable, [sys.executable] + sys.argv)
    time.sleep(0.1)

T = TypeVar("T")
def load_pickle(path:pathlib.Path, default:T) -> T:
  if not path.exists(): return default
  with path.open("rb") as f: return pickle.load(f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--kernels', type=pathlib.Path, help='Path to kernels', default=pathlib.Path(temp("rewrites.pkl", append_user=True)))
  parser.add_argument('--profile', type=pathlib.Path, help='Path to profile', default=pathlib.Path(temp("profile.pkl", append_user=True)))
  args = parser.parse_args()

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    if s.connect_ex(((HOST:="http://127.0.0.1").replace("http://", ""), PORT:=getenv("PORT", 8000))) == 0:
      raise RuntimeError(f"{HOST}:{PORT} is occupied! use PORT= to change.")
  stop_reloader = threading.Event()
  multiprocessing.current_process().name = "VizProcess"    # disallow opening of devices
  st = time.perf_counter()
  print("*** viz is starting")

  ctxs:list[dict] = get_rewrites(trace:=load_pickle(args.kernels, default=RewriteTrace([], [], {})))
  profile_ret = get_profile(load_pickle(args.profile, default=[]))

  server = TCPServerWithReuse(('', PORT), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"), flush=True)
  if len(getenv("BROWSER", "")) > 0: webbrowser.open(f"{HOST}:{PORT}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
