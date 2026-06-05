#!/usr/bin/env python3
import multiprocessing, pickle, difflib, os, threading, json, time, sys, webbrowser, socket, argparse, codecs, io, struct, re, traceback, itertools
import socketserver
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from decimal import Decimal
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler
from typing import Any, TypedDict, TypeVar, Generator, Callable
from tinygrad.helpers import colored, getenv, tqdm, unwrap, word_wrap, TRACEMETA, ProfileEvent, ProfileRangeEvent, TracingKey, ProfilePointEvent, temp
from tinygrad.helpers import printable, Context, START_TIME, NO_COLOR, ansistrip
from tinygrad.renderer.amd.dsl import Inst
from tinygrad.renderer.amd import detect_format

# NOTE: using HTTPServer forces a potentially slow socket.getfqdn
class TCPServerWithReuse(socketserver.TCPServer):
  allow_reuse_address = True
  def __init__(self, server_address, RequestHandlerClass):
    print(f"*** started server on http://127.0.0.1:{server_address[1]} at {time.perf_counter()-START_TIME:.2f} s")
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

from tinygrad.uop.ops import TrackedGraphRewrite, RewriteTrace, UOp, Ops, GroupOp, srender, sint, sym_infer, range_str, range_start, multirange_str
from tinygrad.uop.ops import KernelInfo
from tinygrad.uop.render import print_uops, pyrender
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent, ProfileGraphEntry, ProfileProgramEvent
from tinygrad.dtype import dtypes

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.REDUCE: "#FF5B5B",
               **{x:"#f2cb91" for x in {Ops.DEFINE_LOCAL, Ops.DEFINE_REG}}, Ops.SHAPED_WMMA: "#FF5B5B",
               Ops.RANGE: "#c8a0e0", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#cef263", Ops.WMMA: "#efefc0", Ops.MULTI: "#f6ccff", Ops.INS: "#eec4ff",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80",
               Ops.BUFFER_VIEW: "#E5EAFF", Ops.BUFFER: "#B0BDFF", Ops.GETADDR: "#9DB1F0", Ops.COPY: "#a040a0", Ops.CUSTOM_FUNCTION: "#bf71b6",
               Ops.CALL: "#00B7C8", Ops.FUNCTION: "#C07788", Ops.PARAM: "#14686F", Ops.SOURCE: "#c0c0c0", Ops.BINARY: "#404040",
               Ops.LINEAR: "#7DF4FF",
               Ops.ALLREDUCE: "#ff40a0", Ops.MSELECT: "#d040a0", Ops.MSTACK: "#d040a0", Ops.CONTIGUOUS: "#FFC14D",
               Ops.STAGE: "#AC640D", Ops.REWRITE_ERROR: "#ff2e2e", Ops.AFTER: "#8A7866", Ops.END: "#524C46"}

# VIZ API

# A step is a lightweight descriptor for a trace entry
# Includes a name, metadata and a URL path for fetching the full data

def create_step(name:str, query:tuple[str, int, int], data=None, depth:int=0, **kwargs) -> dict:
  return {"name":name, "query":f"{query[0]}?ctx={query[1]}&step={query[2]}", "data":data, "depth":depth, **kwargs}

@dataclass(frozen=True)
class VizData:
  trace:RewriteTrace = field(default_factory=lambda: RewriteTrace([], [], {}))
  ctxs:list[dict] = field(default_factory=list)
  ref_map:dict[Any, int] = field(default_factory=dict)
  all_uops:dict[int, UOp] = field(default_factory=dict)

# ** load all saved rewrites

def load_rewrites(data:VizData) -> None:
  assert not data.ctxs and not data.ref_map, "load_rewrites called multiple times"
  for i,k in enumerate(data.trace.keys):
    steps:list[dict] = []
    p:UOp|None = None
    for j,s in enumerate(data.trace.rewrites[i]):
      steps.append(create_step(s.name, ("/graph-rewrites", i, j), loc=s.loc, match_count=len(s.matches), code_line=printable(s.loc),
                               trace=k.tb if j==0 else None, depth=s.depth))
      # get source and binary from Ops.PROGRAM
      if s.name == "View Program":
        p = _reconstruct(data, s.sink, depth=1)
        steps.append(create_step("View UOp List", ("/uops", i, len(steps))))
        steps.append(create_step("View Source", ("/code", i, len(steps)), p.src[3].arg))
        steps.append(create_step("View Disassembly", ("/asm", i, len(steps)), (k.ret, p.src[4].arg)))
    for key in k.keys: data.ref_map[canonicalize_ast(key) if isinstance(key, UOp) else key] = i
    data.ctxs.append({"name":k.display_name, "steps":steps, "prg":p})

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

def fmt_colored(s:str) -> str: return ansistrip(s) if NO_COLOR else s

def canonicalize_ast(u:UOp) -> UOp: return u.replace(arg=KernelInfo()) if u.op is Ops.SINK and isinstance(u.arg, KernelInfo) else u

def uop_to_json(data:VizData, x:UOp) -> dict[int, dict]:
  assert isinstance(x, UOp)
  graph: dict[int, dict] = {}
  excluded: set[UOp] = set()
  for u in (toposort:=x.toposort()):
    # always exclude DEVICE/CONST/UNIQUE
    if u.op in {Ops.DEVICE, Ops.CONST, Ops.UNIQUE, Ops.LUNIQUE} and u is not x: excluded.add(u)
    if u.op is Ops.CONST and len(u.src) and u.src[0].op in {Ops.UNIQUE, Ops.LUNIQUE}: excluded.remove(u)
    if u.op is Ops.STACK and len(u.src) == 0: excluded.add(u)
  for u in toposort:
    if u in excluded: continue
    argst = codecs.decode(str(u.arg), "unicode_escape")
    if u.op in GroupOp.Movement: argst = (mask_to_str if u.op in {Ops.SHRINK, Ops.PAD} else shape_to_str)(u.marg)
    if u.op is Ops.BINARY: argst = f"<{len(u.arg)} bytes>"
    wrap_len = 200 if u.op is Ops.SOURCE else 80
    label = f"{str(u.op).split('.')[1]}{(chr(10)+word_wrap(argst.replace(':', ''), wrap=wrap_len)) if u.arg is not None else ''}"
    if u.dtype != dtypes.void: label += f"\n{u.dtype}"
    for idx,x in enumerate(u.src[:1] if u.op in {Ops.STAGE, Ops.INDEX} else (u.src if u.op is not Ops.END else [])):
      if x in excluded:
        # walk through excluded movement ops to find the underlying CONST
        cx = x
        while cx.op in GroupOp.Movement and len(cx.src) >= 1 and cx.src[0] in excluded: cx = cx.src[0]
        arg = f"{cx.arg:g}" if cx.op is Ops.CONST and dtypes.is_float(cx.dtype) else f"{cx.arg}"
        label += f"\n{cx.op.name}{idx} {arg}" + (f" {cx.src[0].op}" if len(cx.src) else "")
    try:
      if len(rngs:=u.ranges):
        label += f"\n({multirange_str(rngs, color=True)})"
      if u._shape is not None:
        label += f"\n{shape_to_str(u.shape)}"
      if u.op in {Ops.CALL, Ops.FUNCTION}:
        label += f"\n{u.src[0].key.hex()[:8]}"
      if u.op in {Ops.INDEX, Ops.STAGE}:
        if len(u.toposort()) < 30: label += f"\n{u.render()}"
        ranges: list[UOp] = []
        for us in u.src[1:]: ranges += [s for s in us.toposort() if s.op in {Ops.RANGE, Ops.SPECIAL}]
        if ranges: label += "\n"+' '.join([f"{s.render()}={s.vmax+1}" for s in ranges])
      if u.op in {Ops.END, Ops.REDUCE} and len(trngs:=list(UOp.sink(*u.src[range_start[u.op]:]).ranges)):
        label += "\n"+' '.join([f"{range_str(s, color=True)}({s.vmax+1})" for s in trngs])
    except Exception:
      label += "\n<ISSUE GETTING LABEL>"
    ref = data.ref_map.get(canonicalize_ast(u.src[0])) if u.op in {Ops.CALL, Ops.FUNCTION} else None
    if ref is not None: label += f"\ncodegen@{fmt_colored(data.ctxs[ref]['name'])}"
    # NOTE: kernel already has metadata in arg
    if TRACEMETA >= 2 and u.metadata is not None and u.op not in {Ops.CALL, Ops.FUNCTION}: label += "\n"+str(u.metadata)
    # limit SOURCE labels line count
    if u.op is Ops.SOURCE and len(lines:=label.split("\n")) > 40:
      label = "\n".join(lines[:30]) + "\n..."
    graph[id(u)] = {"label":label, "src":[(i,id(x)) for i,x in enumerate(u.src) if x not in excluded], "color":uops_colors.get(u.op, "#ffffff"),
                    "ref":ref, "tag":repr(u.tag) if u.tag is not None else None}
  return graph

def _reconstruct(data:VizData, a:int, depth:int|None=None):
  if depth is None and a in data.all_uops: return data.all_uops[a]
  op, dtype, src, arg, *rest = data.trace.uop_fields[a]
  if depth is not None and depth <= 0: return UOp(op, dtype, (), arg, *rest)
  ret = UOp(op, dtype, tuple(_reconstruct(data, s, None if depth is None else depth-1) for s in src), arg, *rest)
  if depth is None: data.all_uops[a] = ret
  return ret

def get_full_rewrite(data:VizData, ctx:TrackedGraphRewrite) -> Generator[GraphRewriteDetails, None, None]:
  next_sink = _reconstruct(data, ctx.sink)
  yield {"graph":uop_to_json(data, next_sink), "uop":pystr(next_sink), "change":None, "diff":None, "upat":None}
  replaces: dict[UOp, UOp] = {}
  for u0_num,u1_num,upat_loc,dur in tqdm(ctx.matches, disable=not ctx.matches):
    replaces[u0:=_reconstruct(data, u0_num)] = u1 = _reconstruct(data, u1_num)
    try: new_sink = next_sink.substitute(replaces)
    except RuntimeError as e: new_sink = UOp(Ops.NOOP, arg=str(e))
    match_repr = f"# {dur*1e6:.2f} us\n"+printable(upat_loc)
    yield {"graph":(sink_json:=uop_to_json(data, new_sink)), "uop":pystr(new_sink), "change":[id(x) for x in u1.toposort() if id(x) in sink_json],
           "diff":list(difflib.unified_diff(pystr(u0).splitlines(), pystr(u1).splitlines())), "upat":(upat_loc, match_repr)}
    if not ctx.bottom_up: next_sink = new_sink

# encoder helpers

def enum_str(s, cache:dict[str, int]) -> int:
  if (cret:=cache.get(s)) is not None: return cret
  cache[s] = ret = len(cache)
  return ret

def option(s:int|None) -> int: return 0 if s is None else s+1

def rel_ts(ts:int|Decimal, start_ts:int, ctx:str="") -> int:
  val = int(ts) - start_ts
  if val < 0 or val > 0xFFFFFFFF: raise ValueError(f"timestamp out of range: {ctx} diff={val} (ts={ts} start={start_ts})")
  return val

# Profiler API

def cpu_ts_diff(device_ts_diffs:dict[str, Decimal], device:str) -> Decimal: return device_ts_diffs.get(device, Decimal(0))

DevEvent = ProfileRangeEvent|ProfileGraphEntry|ProfilePointEvent
def flatten_events(profile:list[ProfileEvent], device_ts_diffs:dict[str, Decimal]) -> Generator[tuple[Decimal, Decimal, DevEvent], None, None]:
  for e in profile:
    if isinstance(e, ProfileRangeEvent): yield (e.st+(diff:=cpu_ts_diff(device_ts_diffs, e.device)), (e.en if e.en is not None else e.st)+diff, e)
    elif isinstance(e, ProfilePointEvent): yield (e.ts, e.ts, e)
    elif isinstance(e, ProfileGraphEvent):
      cpu_ts = []
      for ent in e.ents: cpu_ts += [e.sigs[ent.st_id]+(diff:=cpu_ts_diff(device_ts_diffs, ent.device)), e.sigs[ent.en_id]+diff]
      yield (st:=min(cpu_ts)), (et:=max(cpu_ts)), ProfileRangeEvent(f"{e.ents[0].device.split(':')[0]} Graph", f"batched {len(e.ents)}", st, et)
      for i,ent in enumerate(e.ents): yield (cpu_ts[i*2], cpu_ts[i*2+1], ent)

# normalize event timestamps and attach kernel metadata
def timeline_layout(data:VizData, dev_events:list[tuple[int, int, float, DevEvent]], start_ts:int, scache:dict[str, int]) -> bytes|None:
  events:list[bytes] = []
  ei:ProfilePointEvent|None = None
  for st,et,dur,e in dev_events:
    if isinstance(e, ProfilePointEvent) and e.name == "exec": ei = e
    if dur == 0: continue
    name, key = e.name, None
    fmt:dict = {}
    if (ref:=data.ref_map.get(name)) is not None and ref < len(data.ctxs):
      name = data.ctxs[ref]["name"]
      if (p:=data.ctxs[ref].get("prg")) is not None and (ki:=p.src[0].arg).estimates is not None and ei is not None:
        fmt["FLOPS"] = int(sym_infer(ki.estimates.ops, var_vals:=ei.arg['var_vals'])/(t:=dur*1e-6))
        fmt["B/s mem"], fmt["B/s lds"] = int(sym_infer(ki.estimates.mem, var_vals)/t), int(sym_infer(ki.estimates.lds, var_vals)/t)
        if ei.arg["metadata"]: fmt["metadata"] = ",".join([str(m) for m in ei.arg['metadata']+["batched" if isinstance(e,ProfileGraphEntry) else ""]])
        key = ei.key
    elif isinstance(e.name, TracingKey):
      name = e.name.display_name
      ref = next((v for k in e.name.keys if (v:=data.ref_map.get(k)) is not None), None)
      if isinstance(e.name.ret, str): fmt.update(json.loads(e.name.ret[4:]) if e.name.ret.startswith("JSON") else {"metadata":e.name.ret})
      elif isinstance(e.name.ret, int): fmt["B/s"], fmt["B"] = int(e.name.ret/(dur*1e-6)), e.name.ret
      elif e.name.tb: fmt["tb"] = e.name.tb
    events.append(struct.pack("<IIIIfI", enum_str(name, scache), option(ref), option(key), rel_ts(st,start_ts, f"'{name}' on {e.device}"),
                              dur, enum_str(json.dumps(fmt),scache)))
  return struct.pack("<BI", 0, len(events))+b"".join(events) if events else None

def encode_mem_free(key:int, ts:int, execs:list[ProfilePointEvent], scache:dict) -> bytes:
  ei_encoding:list[tuple[int, int, int, int]] = [] # <[u32, u32, u32, u8] [run id, display name, buffer number and mode (2 = r/w, 1 = w, 0 = r)]
  for e in execs:
    num = next(i for i,k in enumerate(e.arg["bufs"]) if k == key)
    mode = 2 if (num in e.arg["inputs"] and num in e.arg["outputs"]) else 1 if (num in e.arg["outputs"]) else 0
    ei_encoding.append((e.key, enum_str(e.arg["name"], scache), num, mode))
  return struct.pack("<BIII", 0, ts, key, len(ei_encoding))+b"".join(struct.pack("<IIIB", *t) for t in ei_encoding)

def graph_layout(k:str, dev_events:list[tuple[int, int, float, DevEvent]], start_ts:int, end_ts:int, peaks:list[int], dtype_size:dict[str, int],
                 scache:dict[str, int]) -> tuple[str, bytes|None]:
  if k.startswith("LINE:"):
    xy = [(rel_ts(e.ts, start_ts, f"line '{k}' on {e.device}"), e.key) for st,_,_,e in dev_events if isinstance(e, ProfilePointEvent)]
    peaks.append(peak:=max([y for _,y in xy]))
    return k.replace("LINE:", ""), struct.pack("<BIBQ", 1, len(xy), 1, peak)+b"".join(struct.pack("<IQ", x, y) for x,y in xy)
  peak, mem = 0, 0
  temp:dict[int, int] = {}
  events:list[bytes] = []
  buf_ei:dict[int, list[ProfilePointEvent]] = {}
  for st,_,_,e in dev_events:
    if not isinstance(e, ProfilePointEvent): continue
    if e.name == "alloc":
      safe_sz = min(1_000_000_000_000, e.arg["sz"])
      events.append(struct.pack("<BIIIQ", 1, rel_ts(e.ts, start_ts, f"alloc on {e.device}"), e.key, enum_str(e.arg["dtype"].name, scache), safe_sz))
      dtype_size.setdefault(e.arg["dtype"].name, e.arg["dtype"].itemsize)
      temp[e.key] = nbytes = safe_sz*e.arg["dtype"].itemsize
      mem += nbytes
      if mem > peak: peak = mem
    if e.name == "exec" and e.arg["bufs"]:
      for b in e.arg["bufs"]: buf_ei.setdefault(b, []).append(e)
    if e.name == "free":
      events.append(encode_mem_free(e.key, rel_ts(e.ts, start_ts, f"free on {e.device}"), buf_ei.pop(e.key, []), scache))
      mem -= temp.pop(e.key)
  for t in temp: events.append(encode_mem_free(t, rel_ts(end_ts, start_ts, f"end_ts for {k}"), buf_ei.pop(t, []), scache))
  peaks.append(peak)
  return f"{k} Memory", struct.pack("<BIBQ", 1, len(events), 0, peak)+b"".join(events) if events else None

# by default, VIZ does not start when there is an error
# use this to instead display the traceback to the user
@contextmanager
def soft_err(fn:Callable):
  try: yield
  except Exception: fn({"src":traceback.format_exc()})

def row_tuple(row:str) -> tuple[tuple[int, int], ...]:
  return ((0, 0),) if "Clock" in row else tuple((ord(ss[0][0]), int(ss[1])) if len(ss:=x.split(":"))>1 else (999,999) for x in row.split())

# *** Performance counters

metrics:dict[str, Callable[[dict[str, tuple[int, int, int]]], str]] = {
  "VALU utilization": lambda s: f"{100 * (s['SQ_INSTS_VALU'][0] / s['SQ_INSTS_VALU'][2]) / (s['GRBM_GUI_ACTIVE'][1] * 4):.1f}%",
  "SALU utilization": lambda s: f"{100 * (s['SQ_INSTS_SALU'][0] / s['SQ_INSTS_SALU'][2]) / (s['GRBM_GUI_ACTIVE'][1] * 4):.1f}%",
}

def unpack_pmc(e) -> dict:
  agg_cols = ["Name", "Sum"]
  rows:list[list] = []
  stats:dict[str, tuple[int, int, int]] = {}  # name -> (sum, max, count)
  view, ptr = memoryview(e.blob).cast('Q'), 0
  for s in e.sched:
    sample_cols = ["XCC", "INST", "SE", "SA"] + [f"WGP:{i}" for i in range(s.wgp)]
    row:list = [s.name, 0, {"cols":sample_cols, "rows":[]}]
    max_val, cnt = 0, 0
    for sample in itertools.product(range(s.xcc), range(s.inst), range(s.se), range(s.sa)):
      vals:list[int] = []
      # pack work group processors on the same se
      for _ in range(s.wgp):
        row[1] += (val:=int(view[ptr]))
        max_val, cnt = max(max_val, val), cnt + 1
        vals.append(val)
        ptr += 1
      row[2]["rows"].append(sample+tuple(vals))
    stats[s.name] = (row[1], max_val, cnt)
    rows.append(row)
  for name, fn in metrics.items():
    try: rows.append([name, fn(stats)])
    except KeyError: pass
  return {"rows":rows, "cols":agg_cols}

# ** on startup, list all the performance counter traces

def load_amd_counters(data:VizData, profile:list) -> None:
  counter_events:dict[tuple[int, int], dict] = {}
  durations:dict[str, list[float]] = {}
  prg_events:dict[int, ProfileProgramEvent] = {}
  arch = ""
  for e in profile:
    if type(e).__name__ in {"ProfilePMCEvent", "ProfileSQTTEvent"}:
      counter_events.setdefault((e.kern, e.exec_tag), {}).setdefault(type(e).__name__, []).append(e)
    if isinstance(e, ProfileRangeEvent) and e.device.startswith("AMD") and e.en is not None:
      durations.setdefault(str(e.name), []).append(float(e.en-e.st))
    if isinstance(e, ProfileProgramEvent) and e.tag is not None: prg_events[e.tag] = e
    if isinstance(e, ProfileDeviceEvent) and e.device.startswith("AMD"): arch = f"gfx{unwrap(e.props)['gfx_target_version']//1000}"
  if len(counter_events) == 0: return None
  data.ctxs.append({"name":"All Counters", "steps":[create_step("PMC", ("/all-pmc", len(data.ctxs), 0), (durations, all_counters:={}))]})
  run_number = {n:0 for n,_ in counter_events}
  for (k, tag),v in counter_events.items():
    # use the colored name if it exists
    name = data.ctxs[r]["prg"].src[0].arg.name if (r:=data.ref_map.get(pname:=prg_events[k].name)) is not None else pname
    run_number[k] += 1
    steps:list[dict] = []
    if (pmc:=v.get("ProfilePMCEvent")):
      steps.append(create_step("PMC", ("/prg-pmc", len(data.ctxs), len(steps)), pmc[0]))
      all_counters[(name, run_number[k], pname)] = pmc[0]
    # to decode a SQTT trace, we need the raw stream, program binary and device properties
    if (sqtt:=v.get("ProfileSQTTEvent")):
      for e in sqtt:
        if e.itrace: steps.append(create_step(f"SE:{e.se} PKTS", (f"/sqtt-{e.se}",len(data.ctxs),len(steps)), data=(e.blob,prg_events[k].lib,arch)))
      try:
        with Context(DEBUG=0): from extra.sqtt.roc import unpack_occ
        steps.append(create_step("OCC", ("/amd-sqtt-occ", len(data.ctxs), len(steps)),
                                 data={"fxn":unpack_occ, "args":((k, tag), sqtt, prg_events[k], arch)}))
      except Exception: pass
    data.ctxs.append({"name":f"SQTT {name}"+(f" n{run_number[k]}" if run_number[k] > 1 else ""), "steps":steps})

wave_colors = {"WMMA": "#1F7857", **{x:"#ffffc0" for x in ["VALU", "VINTERP"]}, "SALU": "#cef263", "SMEM": "#ffc0c0", "STORE": "#4fa3cc",
               **{x:"#b2b7c9" for x in ["VMEM", "SGMEM"]}, "LDS": "#9fb4a6", "IMMEDIATE": "#f3b44a", "BARRIER": "#d00000",
               "JUMP_NO": "#fb8500", "JUMP": "#ffb703", "WAVERDY": "#1a2a2a"}

def sqtt_timeline(data:bytes, lib:bytes, target:str) -> Generator[ProfileEvent, None, None]:
  from tinygrad.renderer.amd.sqtt import (map_insts, InstructionInfo, PacketType, INST, InstOp, VALUINST, IMMEDIATE, IMMEDIATE_MASK, VMEMEXEC,
                                          ALUEXEC, INST_RDNA4, InstOpRDNA4, TS_DELTA_OR_MARK, TS_DELTA_OR_MARK_RDNA4, CDNA_INST, InstOpCDNA,
                                          WAVEEND, WAVEEND_RDNA4, CDNA_WAVEEND, WAVERDY)
  pc_map = {addr:str(inst) for addr,inst in amd_decode(lib, target).items()}
  row_ends:dict[str, Decimal] = {}
  row_counts:dict[str, itertools.count] = {}
  curr_barrier:dict[int, ProfileRangeEvent] = {}
  exec_pending:dict[str, list[tuple[str, str]]] = {}
  dispatch_to_exec = {"WMMA":"VALU", "VALU":"VALU", "VALU1":"VALU", "VALUT":"VALU", "VALUB":"VALU", "VALUINST":"VALU", "VINTERP":"VALU",
                      "SGMEM":"VMEM", "FLAT":"VMEM", "LDS":"LDS", "SALU":"SALU", "SMEM":"SALU", "VMEM":"VMEM"}
  def add(name:str, p:PacketType, wave:int|None=None, info:InstructionInfo|None=None) -> Generator[ProfileEvent, None, None]:
    row = f"WAVE:{wave}" if (wave:=getattr(p, "wave", wave)) is not None else f"{p.__class__.__name__}:0 {name.replace('_ALT', '')}"
    # by default we extend the packet to one cycle after timestamp
    start_time, end_time = p._time, p._time+1
    # exec links to dispatch, dispatch links to PC
    link:dict|None = {"pc":info.pc} if info else None
    if isinstance(p, (ALUEXEC, VMEMEXEC)):
      dispatch_id, op_type = exec_pending[name].pop(0)
      # wmma exec gets its own color and its own row on rdna4
      if op_type.startswith("WMMA"):
        name = name+"_WMMA"
        if not op_type.startswith("WMMA_VALU"): row = "ALUEXEC:0 WMMA"
      # transcendental valu gets its own row
      if op_type.startswith("VALUT"): row = "ALUEXEC:0 TFU"
      # extend execs by the op type's known duration, p._time marks the first or last cycle based on the op type
      duration = int(dur_match.group(1)) if (dur_match:=re.match(r".*_(\d+)$", op_type)) else 1
      if any(ss in row for ss in ("SALU", "TFU", "VMEM", "LDS")): start_time, end_time = p._time, p._time+duration
      else: start_time, end_time = p._time-duration, p._time
      link = {"link":dispatch_id}
    # queue inst dispatches
    idx = next(row_counts.setdefault(row, itertools.count(0)))
    if isinstance(p, (VALUINST, INST, INST_RDNA4)) and (exec_type:=dispatch_to_exec.get(name.replace("OTHER_", "").split("_")[0])) is not None:
      if name.startswith("OTHER_"): exec_type = f"{exec_type}_ALT"
      # detect rdna3 wmma from the asm, only rdna4 has an op type for it
      if isinstance(p, VALUINST) and (asm:=getattr(unwrap(info).inst, "op_name", "")).startswith("V_WMMA"):
        name = f"WMMA_VALU_{16 if 'IU4' in asm else 32}"
      exec_pending.setdefault(exec_type, []).append((f"{row}-{idx}", name))
    # construct and yield the event for this packet
    if row not in row_ends: yield ProfilePointEvent(row, "JSON", "pcMap", pc_map, ts=Decimal(0))
    yield (e:=ProfileRangeEvent(row, TracingKey(name, ret="JSON"+json.dumps(link) if link else None), Decimal(start_time), Decimal(end_time)))
    row_ends[row] = unwrap(e.en)
    # barrier on this wave extends to fill the time it was waiting
    if wave is not None:
      if (barrier:=curr_barrier.pop(wave, None)) is not None: barrier.en = Decimal(p._time)
      if name in {"BARRIER", "BARRIER_SIGNAL"}: curr_barrier[wave] = e
  NS_PER_TICK = 10  # 100MHz
  prev_pair:tuple[int, int]|None = None # (shader, realtime)
  yield ProfilePointEvent("", "JSON", "waveColors", list(wave_colors.items()), ts=Decimal(0))
  for p, info in map_insts(data, lib, target):
    if isinstance(p, (TS_DELTA_OR_MARK, TS_DELTA_OR_MARK_RDNA4)) and p.is_marker:
      pair = (p._time, p.delta)
      if prev_pair is None: prev_pair = pair
      else:
        (s0, r0), (s1, r1) = prev_pair, pair
        freq_hz = (s1 - s0) * 1_000_000_000 // ((r1 - r0) * NS_PER_TICK)
        yield ProfilePointEvent("LINE:Shader Clock", "freq_hz", freq_hz, ts=Decimal(p._time))
        prev_pair = pair
    if isinstance(p, (INST, INST_RDNA4, CDNA_INST)):
      name = p.op.name if isinstance(p.op, (InstOp, InstOpRDNA4, InstOpCDNA)) else f"0x{p.op:02x}"
      yield from add(name, p, info=info)
    if isinstance(p, (VALUINST, IMMEDIATE, WAVEEND, WAVEEND_RDNA4, CDNA_WAVEEND)): yield from add(p.__class__.__name__, p, info=info)
    if isinstance(p, IMMEDIATE_MASK): yield from add("IMMEDIATE", p, wave=unwrap(info).wave, info=info)
    if isinstance(p, WAVERDY):
      for wave in range(16):
        if p.mask & (1 << wave):
          if wave in curr_barrier: yield from add("WAVERDY", p, wave=wave)
    if isinstance(p, (VMEMEXEC, ALUEXEC)):
      name = str(p.src).split('.')[1]
      if name == "VALU_SALU":
        yield from add("VALU", p)
        yield from add("SALU", p)
      else:
        yield from add(name, p)

def device_sort_fn(k:str) -> tuple:
  special = {"GC": 0, "USER": 1, "TINY": 2, "ALLDEVS":100, "DISK": 999}
  is_memory = k.endswith(" Memory")
  p = k.split(" ")[0].split(":")
  dev_base = p[0] if len(p) < 2 or not p[1].isdigit() else f"{p[0]}:{p[1]}"
  return (is_memory, special.get(p[0], special['ALLDEVS']), dev_base, k)

def get_profile(data:VizData, profile:list[ProfileEvent], sort_fn:Callable[[str], Any]=device_sort_fn) -> bytes|None:
  # start by getting the time diffs
  device_ts_diffs:dict[str, Decimal] = {}
  device_decoders:dict[str, Callable[[VizData, list[ProfileEvent]], None]] = {}
  for ev in profile:
    if isinstance(ev, ProfileDeviceEvent):
      device_ts_diffs[ev.device] = ev.tdiff
      if (d:=ev.device.split(":")[0]) == "AMD": device_decoders[d] = load_amd_counters
      if d == "NV": device_decoders[d] = load_nv_counters
  # load device specific counters
  for fxn in device_decoders.values(): fxn(data, profile)
  # map events per device
  dev_events:dict[str, list[tuple[int, int, float, DevEvent]]] = {}
  markers:list[ProfilePointEvent] = []
  ext_data:dict[str, Any] = {}
  start_ts:int|None = None
  end_ts:int|None = None
  for ts,en,e in flatten_events(profile, device_ts_diffs):
    dev_events.setdefault(e.device,[]).append((st:=int(ts), et:=int(en), float(en-ts), e))
    if start_ts is None or st < start_ts: start_ts = st
    if end_ts is None or et > end_ts: end_ts = et
    if isinstance(e, ProfilePointEvent) and e.name == "marker": markers.append(e)
    if isinstance(e, ProfilePointEvent) and e.name == "JSON": ext_data[e.key] = e.arg
  if start_ts is None: return None
  # return layout of per device events
  layout:dict[str, bytes|None] = {}
  scache:dict[str, int] = {}
  peaks:list[int] = []
  dtype_size:dict[str, int] = {}
  for k,v in dev_events.items():
    v.sort(key=lambda e:e[0])
    layout[k] = timeline_layout(data, v, start_ts, scache)
    layout.update([graph_layout(k, v, start_ts, unwrap(end_ts), peaks, dtype_size, scache)])
  sorted_layout = sorted([k for k,v in layout.items() if v is not None], key=sort_fn)
  ret = [b"".join([struct.pack("<B", len(k)), k.encode(), unwrap(layout[k])]) for k in sorted_layout]
  index = json.dumps({"strings":list(scache), "dtypeSize":dtype_size,
                      "markers":[{"ts":rel_ts(e.ts, start_ts, f"marker '{e.arg.get('name','?')}'"), **e.arg} for e in markers],
                      **ext_data}).encode()
  return struct.pack("<IQII", rel_ts(unwrap(end_ts), start_ts, "end_ts"), max(peaks,default=0), len(index), len(ret))+index+b"".join(ret)

# ** PMA counters

def load_nv_counters(data:VizData, profile:list) -> None:
  steps:list[dict] = []
  sm_version = {e.device:e.props.get("sm_version", 0x800) for e in profile if isinstance(e, ProfileDeviceEvent) and e.props is not None}
  run_number:dict[str, int] = {}
  for e in profile:
    if type(e).__name__ == "ProfilePMAEvent":
      run_number[e.kern] = run_num = run_number.get(e.kern, 0)+1
      steps.append(create_step(f"PMA {e.kern}"+(f"n{run_num}" if run_num>1 else ""), ("/prg-pma-pkts", len(data.ctxs), len(steps)),
                               data=(e.blob, sm_version[e.device])))
  if steps: data.ctxs.append({"name":"All Counters", "steps":steps})

def pma_timeline(blob:bytes, sm_version:int) -> list[ProfileEvent]:
  from extra.nv_pma.decode import decode, decode_tpc_id
  ret:list[ProfileEvent] = []
  rows:dict[str, None] = {}
  tpc_count:dict[int, int] = {}
  # assume every sample is 32 cycles
  cycles_per_sample = 32
  for s, tpc_id in decode(blob, sm_version):
    if len(ret) > getenv("MAX_SQTT_PKTS", 50_000): break
    gpc, tpc, sm = decode_tpc_id(tpc_id)
    tpc_count[tpc_id] = (n:=tpc_count.get(tpc_id,0)) + 1
    rows.setdefault(row:=f"GPC:{gpc} TPC:{tpc} SM:{sm} WAVE:{s.wave_id}")
    ret.append(ProfileRangeEvent(row, TracingKey(s.stall_reason.name, ret=f"pc=0x{s.pc_offset:06x} active={s.active}"),
                                 Decimal(n*cycles_per_sample), Decimal((n+1)*cycles_per_sample)))
  return [ProfilePointEvent(r, "start", r, ts=Decimal(0)) for r in rows]+ret

# ** Assembly static analyzers

def get_stdout(f: Callable) -> str:
  buf = io.StringIO()
  try:
    with redirect_stdout(buf), redirect_stderr(buf): f()
  except Exception: traceback.print_exc(file=buf)
  return buf.getvalue()

def get_elf_section(lib:bytes, name:str):
  from tinygrad.runtime.support.elf import elf_loader
  return next((sh for sh in elf_loader(lib)[1] if sh.name == name))

def amd_decode(lib:bytes, target:str) -> dict[int, Inst]:
  text = get_elf_section(lib, ".text")
  off, buf = text.header.sh_addr, text.content
  arch = "rdna3" if target.startswith("gfx11") else "rdna4" if target.startswith("gfx12") else "cdna"
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
def amdgpu_cfg(lib:bytes, target:str) -> dict:
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
  for pc, inst in pc_table.items():
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
  from tinygrad.renderer.amd.dsl import Reg
  for pc, inst in pc_table.items():
    pc_tokens[pc] = tokens = []
    for name, f in inst._fields:
      if isinstance(val:=getattr(inst, name), Reg): tokens.append({"st":val.fmt(), "keys":[f"r{val.offset+i}" for i in range(val.sz)], "kind":1})
      elif name in {"op","opx","opy"}: tokens.append({"st":(op_name:=val.name.lower()), "keys":[op_name], "kind":0})
      elif name != "encoding" and val != f.default: tokens.append({"st":(s:=repr(val)), "keys":[s], "kind":1})
  # show a smaller view for repeated instructions in the graph
  lines:list[str] = []
  disasm = {pc:str(inst) for pc,inst in pc_table.items()}
  asm_width = max(len(asm) for asm in disasm.values())
  for pcs in blocks.values():
    new_pcs:list[int] = []
    i, n = 0, len(pcs)
    while i < n:
      j = i+1
      while j<n and pc_table[pcs[j]] == pc_table[pcs[i]]: j += 1
      new_pcs.append(pcs[i])
      if j-i>1:
        pc_tokens[pcs[i]].append({"st":f"({j-i}x)", "keys":[], "kind":0})
        for k in range(i+1, j): del pc_tokens[pcs[k]]
      lines.append(f"{disasm[pcs[i]]:<{asm_width}}  # {pcs[i]:012X}"+(f"...{pcs[j-1]:012X} ({j-i}x)" if j-i>1 else ""))
      i = j
    pcs[:] = new_pcs
  from tinygrad.runtime.autogen import amdgpu_kd
  kd = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytearray(get_elf_section(lib, ".rodata").content))
  vgpr_gran = kd.compute_pgm_rsrc1 & amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT
  return {"data":{"blocks":blocks, "paths":paths, "pc_tokens":pc_tokens}, "src":"\n".join(lines), "lang":"python",
          "metadata":[[{"label":f"{r} Alloc", "value":v} for r,v in [("VGPR", (vgpr_gran+1)*8-7), ("LDS", kd.group_segment_fixed_size),
                                                                     ("Scratch", kd.private_segment_fixed_size)] if v>0]]}

# ** Main render function to get the complete details about a trace event

def get_render(viz_data:VizData, query:str) -> dict:
  url = urlparse(query)
  i, j, fmt = get_int(qs:=parse_qs(url.query), "ctx"), get_int(qs, "step"), url.path.lstrip("/")
  data = viz_data.ctxs[i]["steps"][j]["data"]
  if fmt == "graph-rewrites": return {"value":get_full_rewrite(viz_data, viz_data.trace.rewrites[i][j]), "content_type":"text/event-stream"}
  if fmt == "uops": return {"src":get_stdout(lambda: print_uops(_reconstruct(viz_data, viz_data.trace.rewrites[i][j-1].sink).src[2].src))}
  if fmt == "code": return {"src":data, "lang":"cpp"}
  if fmt == "asm":
    ret:dict = {}
    renderer, lib = data
    if renderer.target.arch.startswith("gfx"):
      with soft_err(lambda err: ret.update(err)): ret.update(amdgpu_cfg(lib, renderer.target.arch))
    else: ret["src"] = get_stdout(lambda: renderer.compiler.disassemble(lib))
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
  if fmt == "prg-pmc": return unpack_pmc(data)
  if fmt.startswith("sqtt"):
    ret = {}
    with soft_err(lambda err:ret.update(err)):
      if (events:=get_profile(viz_data, list(itertools.islice(sqtt_timeline(*data), getenv("MAX_SQTT_PKTS", 50_000))), sort_fn=row_tuple)):
        ret = {"value":events, "content_type":"application/octet-stream"}
      else: ret = {"src":"No SQTT trace on this SE."}
    return ret
  # viewers for the amd decoder in extra
  if fmt.startswith("amd-sqtt"): return data["fxn"](viz_data, i, j, *data["args"])
  if fmt == "cu-sqtt": return {"value":get_profile(viz_data, data, sort_fn=row_tuple), "content_type":"application/octet-stream"}
  if fmt == "prg-pma-pkts":
    ret = {}
    with soft_err(lambda err:ret.update(err)):
      if (events:=get_profile(viz_data, pma_timeline(*data), sort_fn=row_tuple)): ret = {"value":events, "content_type":"application/octet-stream"}
      else: ret = {"src":"No PMA samples found."}
    return ret
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
      lst = [{"name":c["name"], "steps":[{k:v for k, v in s.items() if k != "data"} for s in c["steps"]]} for c in data.ctxs]
      ret, content_type = json.dumps(lst).encode(), "application/json"
    elif url.path == "/get_profile" and profile_ret: ret, content_type = profile_ret, "application/octet-stream"
    else:
      if not (render_src:=get_render(data, self.path)): status_code = 404
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
# unpickling may load libraries, turn off DEBUG=3 output
@Context(DEBUG=0)
def load_pickle(path:str, default:T) -> T:
  if not os.path.exists(path): return default
  with open(path, "rb") as f: return pickle.load(f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--rewrites-path', type=str, help='Path to rewrites', default=temp("rewrites.pkl", append_user=True))
  parser.add_argument('--profile-path', type=str, help='Path to profile', default=temp("profile.pkl", append_user=True))
  args = parser.parse_args()

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    if s.connect_ex(((HOST:="http://127.0.0.1").replace("http://", ""), PORT:=getenv("PORT", 8000))) == 0:
      raise RuntimeError(f"{HOST}:{PORT} is occupied! use PORT= to change.")
  stop_reloader = threading.Event()
  multiprocessing.current_process().name = "VizProcess"
  Context(ALLOW_DEVICE_USAGE=0).__enter__()                # disallow opening of devices
  st = time.perf_counter()
  print("*** viz is starting")

  data = VizData(load_pickle(args.rewrites_path, default=RewriteTrace([], [], {})))
  load_rewrites(data)
  profile_ret = get_profile(data, load_pickle(args.profile_path, default=[]))

  server = TCPServerWithReuse(('', PORT), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"), flush=True)
  if len(getenv("BROWSER", "")) > 0: webbrowser.open(f"{HOST}:{PORT}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
