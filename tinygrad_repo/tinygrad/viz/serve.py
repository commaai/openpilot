#!/usr/bin/env python3
import multiprocessing, pickle, difflib, os, threading, json, time, sys, webbrowser, socket, argparse, socketserver, functools, codecs, io, struct
import subprocess, ctypes, pathlib, traceback
from contextlib import redirect_stdout, redirect_stderr
from decimal import Decimal
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from typing import Any, TypedDict, TypeVar, Generator, Callable
from tinygrad.helpers import colored, getenv, tqdm, unwrap, word_wrap, TRACEMETA, ProfileEvent, ProfileRangeEvent, TracingKey, ProfilePointEvent, temp
from tinygrad.helpers import printable
from tinygrad.uop.ops import TrackedGraphRewrite, RewriteTrace, UOp, Ops, GroupOp, srender, sint, sym_infer, range_str, pyrender
from tinygrad.uop.ops import print_uops, range_start, multirange_str
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent, ProfileGraphEntry, Device
from tinygrad.renderer import ProgramSpec
from tinygrad.dtype import dtypes

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.VCONST: "#e0e0e0", Ops.REDUCE: "#FF5B5B",
               Ops.DEFINE_GLOBAL:"#cb9037", **{x:"#f2cb91" for x in {Ops.DEFINE_LOCAL, Ops.DEFINE_REG}}, Ops.REDUCE_AXIS: "#FF6B6B",
               Ops.RANGE: "#c8a0e0", Ops.ASSIGN: "#909090", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#cef263", Ops.WMMA: "#efefc0", Ops.MULTI: "#f6ccff", Ops.KERNEL: "#3e7f55",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80",
               Ops.BUFFER_VIEW: "#E5EAFF", Ops.BUFFER: "#B0BDFF", Ops.COPY: "#a040a0",
               Ops.ALLREDUCE: "#ff40a0", Ops.MSELECT: "#d040a0", Ops.MSTACK: "#d040a0", Ops.CONTIGUOUS: "#FFC14D",
               Ops.BUFFERIZE: "#FF991C", Ops.REWRITE_ERROR: "#ff2e2e", Ops.AFTER: "#8A7866", Ops.END: "#524C46"}

# VIZ API

# ** list all saved rewrites

ref_map:dict[Any, int] = {}
def get_rewrites(t:RewriteTrace) -> list[dict]:
  ret = []
  for i,(k,v) in enumerate(zip(t.keys, t.rewrites)):
    steps = [{"name":s.name, "loc":s.loc, "match_count":len(s.matches), "code_line":printable(s.loc), "trace":k.tb if j == 0 else None,
              "query":f"/ctxs?ctx={i}&idx={j}", "depth":s.depth} for j,s in enumerate(v)]
    if isinstance(k.ret, ProgramSpec):
      steps.append({"name":"View UOp List", "query":f"/render?ctx={i}&fmt=uops", "depth":0})
      steps.append({"name":"View Program", "query":f"/render?ctx={i}&fmt=src", "depth":0})
      steps.append({"name":"View Disassembly", "query":f"/render?ctx={i}&fmt=asm", "depth":0})
    for key in k.keys: ref_map[key] = i
    ret.append({"name":k.display_name, "steps":steps})
  return ret

# ** get the complete UOp graphs for one rewrite

class GraphRewriteDetails(TypedDict):
  graph: dict                            # JSON serialized UOp for this rewrite step
  uop: str                               # strigified UOp for this rewrite step
  diff: list[str]|None                   # diff of the single UOp that changed
  changed_nodes: list[int]|None          # the changed UOp id + all its parents ids
  upat: tuple[tuple[str, int], str]|None # [loc, source_code] of the matched UPat

def shape_to_str(s:tuple[sint, ...]): return "(" + ','.join(srender(x) for x in s) + ")"
def mask_to_str(s:tuple[tuple[sint, sint], ...]): return "(" + ','.join(shape_to_str(x) for x in s) + ")"
def pystr(u:UOp, i:int) -> str:
  try: return pyrender(u)
  except Exception: return str(u)

def uop_to_json(x:UOp) -> dict[int, dict]:
  assert isinstance(x, UOp)
  graph: dict[int, dict] = {}
  excluded: set[UOp] = set()
  for u in (toposort:=x.toposort()):
    # always exclude DEVICE/CONST/UNIQUE
    if u.op in {Ops.DEVICE, Ops.CONST, Ops.UNIQUE} and u is not x: excluded.add(u)
    if u.op is Ops.VCONST and u.dtype.scalar() == dtypes.index and u is not x: excluded.add(u)
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
        label += f"\n{u.render()}"
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

def get_full_rewrite(ctx:TrackedGraphRewrite, i:int=0) -> Generator[GraphRewriteDetails, None, None]:
  next_sink = _reconstruct(ctx.sink)
  # in the schedule graph we don't show indexing ops (unless it's in a kernel AST or rewriting dtypes.index sink)
  yield {"graph":uop_to_json(next_sink), "uop":pystr(next_sink,i), "changed_nodes":None, "diff":None, "upat":None}
  replaces: dict[UOp, UOp] = {}
  for u0_num,u1_num,upat_loc,dur in tqdm(ctx.matches):
    replaces[u0:=_reconstruct(u0_num)] = u1 = _reconstruct(u1_num)
    try: new_sink = next_sink.substitute(replaces)
    except RuntimeError as e: new_sink = UOp(Ops.NOOP, arg=str(e))
    match_repr = f"# {dur*1e6:.2f} us\n"+printable(upat_loc)
    yield {"graph":(sink_json:=uop_to_json(new_sink)), "uop":pystr(new_sink,i),
           "changed_nodes":[id(x) for x in u1.toposort() if id(x) in sink_json],
           "diff":list(difflib.unified_diff(pystr(u0,i).splitlines(),pystr(u1,i).splitlines())), "upat":(upat_loc, match_repr)}
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
    events.append(struct.pack("<IIIIfI", enum_str(name, scache), option(ref), option(key), st-start_ts, dur, enum_str("\n".join(fmt), scache)))
  return struct.pack("<BI", 0, len(events))+b"".join(events) if events else None

def encode_mem_free(key:int, ts:int, execs:list[ProfilePointEvent], scache:dict) -> bytes:
  ei_encoding:list[tuple[int, int, int, int]] = [] # <[u32, u32, u8, u8] [run id, display name, buffer number and mode (2 = r/w, 1 = w, 0 = r)]
  for e in execs:
    num = next(i for i,k in enumerate(e.arg["bufs"]) if k == key)
    mode = 2 if (num in e.arg["inputs"] and num in e.arg["outputs"]) else 1 if (num in e.arg["outputs"]) else 0
    ei_encoding.append((e.key, enum_str(e.arg["name"], scache), num, mode))
  return struct.pack("<BIII", 0, ts, key, len(ei_encoding))+b"".join(struct.pack("<IIBB", *t) for t in ei_encoding)

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

def load_sqtt(profile:list[ProfileEvent]) -> None:
  from tinygrad.runtime.ops_amd import ProfileSQTTEvent
  if not (sqtt_events:=[e for e in profile if isinstance(e, ProfileSQTTEvent)]): return None
  def err(name:str, msg:str|None=None) -> None:
    step = {"name":name, "data":{"src":msg or traceback.format_exc()}, "depth":0, "query":f"/render?ctx={len(ctxs)}&step=0&fmt=counters"}
    return ctxs.append({"name":"Counters", "steps":[step]})
  try: from extra.sqtt.roc import decode
  except Exception: return err("DECODER IMPORT ISSUE")
  try: rctx = decode(profile)
  except Exception: return err("DECODER ERROR")
  if not rctx.inst_execs: return err("EMPTY SQTT OUTPUT", f"{len(sqtt_events)} SQTT events recorded, none got decoded")
  steps:list[dict] = []
  for name,waves in rctx.inst_execs.items():
    if (r:=ref_map.get(name)): name = ctxs[r]["name"]
    steps.append({"name":name, "depth":0, "query":f"/render?ctx={len(ctxs)}&step={len(steps)}&fmt=counters",
                  "data":{"src":trace.keys[r].ret.src if r else name, "lang":"cpp"}})

    # Idle:     The total time gap between the completion of previous instruction and the beginning of the current instruction.
    #           The idle time can be caused by:
    #             * Arbiter loss
    #             * Source or destination register dependency
    #             * Instruction cache miss
    # Stall:    The total number of cycles the hardware pipe couldn't issue an instruction.
    # Duration: Total latency in cycles, defined as "Stall time + Issue time" for gfx9 or "Stall time + Execute time" for gfx10+.
    for w in waves:
      rows, prev_instr = [], w.begin_time
      for i,e in enumerate(w.insts):
        rows.append((e.inst, e.time, max(0, e.time-prev_instr), e.dur, e.stall, str(e.typ).split("_")[-1]))
        prev_instr = max(prev_instr, e.time + e.dur)
      summary = [{"label":"Total Cycles", "value":w.end_time-w.begin_time}, {"label":"CU", "value":w.cu},
                 {"label":"SIMD", "value":w.simd}]
      steps.append({"name":f"Wave {w.wave_id}", "depth":1, "query":f"/render?ctx={len(ctxs)}&step={len(steps)}&fmt=counters",
                    "data":{"rows":rows, "cols":["Instruction", "Clk", "Idle", "Duration", "Stall", "Type"], "summary":summary}})
  ctxs.append({"name":"Counters", "steps":steps})

def get_profile(profile:list[ProfileEvent]) -> bytes|None:
  # start by getting the time diffs
  for ev in profile:
    if isinstance(ev,ProfileDeviceEvent): device_ts_diffs[ev.device] = (ev.comp_tdiff, ev.copy_tdiff if ev.copy_tdiff is not None else ev.comp_tdiff)
  # load device specific counters
  device_decoders:dict[str, Callable[[list[ProfileEvent]], None]] = {}
  for device in device_ts_diffs:
    d = device.split(":")[0]
    if d == "AMD": device_decoders[d] = load_sqtt
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
  groups = sorted(layout.items(), key=lambda x: '' if len(ss:=x[0].split(" ")) == 1 else ss[1])
  ret = [b"".join([struct.pack("<B", len(k)), k.encode(), v]) for k,v in groups if v is not None]
  index = json.dumps({"strings":list(scache), "dtypeSize":dtype_size, "markers":[{"ts":int(e.ts-start_ts), **e.arg} for e in markers]}).encode()
  return struct.pack("<IQII", unwrap(end_ts)-start_ts, max(peaks,default=0), len(index), len(ret))+index+b"".join(ret)

# ** Assembly analyzers

def get_llvm_mca(asm:str, mtriple:str, mcpu:str) -> dict:
  target_args = [f"-mtriple={mtriple}", f"-mcpu={mcpu}"]
  # disassembly output can include headers / metadata, skip if llvm-mca can't parse those lines
  data = json.loads(subprocess.check_output(["llvm-mca","-skip-unsupported-instructions=parse-failure","--json","-"]+target_args, input=asm.encode()))
  cr = data["CodeRegions"][0]
  resource_labels = [repr(x)[1:-1] for x in data["TargetInfo"]["Resources"]]
  rows:list = [[instr] for instr in cr["Instructions"]]
  # add scheduler estimates
  for info in cr["InstructionInfoView"]["InstructionList"]: rows[info["Instruction"]].append(info["Latency"])
  # map per instruction resource usage
  instr_usage:dict[int, dict[int, int]] = {}
  for d in cr["ResourcePressureView"]["ResourcePressureInfo"]:
    instr_usage.setdefault(i:=d["InstructionIndex"], {}).setdefault(r:=d["ResourceIndex"], 0)
    instr_usage[i][r] += d["ResourceUsage"]
  # last row is the usage summary
  summary = [{"idx":k, "label":resource_labels[k], "value":v} for k,v in instr_usage.pop(len(rows), {}).items()]
  max_usage = max([sum(v.values()) for i,v in instr_usage.items() if i<len(rows)], default=0)
  for i,usage in instr_usage.items(): rows[i].append([[k, v, (v/max_usage)*100] for k,v in usage.items()])
  return {"rows":rows, "cols":["Opcode", "Latency", {"title":"HW Resources", "labels":resource_labels}], "summary":summary}

def get_stdout(f: Callable) -> str:
  buf = io.StringIO()
  try:
    with redirect_stdout(buf), redirect_stderr(buf): f()
  except Exception: traceback.print_exc(file=buf)
  return buf.getvalue()

def get_render(i:int, j:int, fmt:str) -> dict|None:
  if fmt == "counters": return ctxs[i]["steps"][j]["data"]
  if not isinstance(prg:=trace.keys[i].ret, ProgramSpec): return None
  if fmt == "uops": return {"src":get_stdout(lambda: print_uops(prg.uops or [])), "lang":"txt"}
  if fmt == "src": return {"src":prg.src, "lang":"cpp"}
  compiler = Device[prg.device].compiler
  disasm_str = get_stdout(lambda: compiler.disassemble(compiler.compile(prg.src)))
  from tinygrad.runtime.support.compiler_cpu import llvm, LLVMCompiler
  if isinstance(compiler, LLVMCompiler):
    mtriple = ctypes.string_at(llvm.LLVMGetTargetMachineTriple(tm:=compiler.target_machine)).decode()
    mcpu = ctypes.string_at(llvm.LLVMGetTargetMachineCPU(tm)).decode()
    ret = get_llvm_mca(disasm_str, mtriple, mcpu)
  else: ret = {"src":disasm_str, "lang":"x86asm"}
  return ret

# ** HTTP server

def get_int(query:dict[str, list[str]], k:str) -> int: return int(query.get(k,["0"])[0])

class Handler(BaseHTTPRequestHandler):
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
    elif (query:=parse_qs(url.query)):
      if url.path == "/render":
        render_src = get_render(get_int(query, "ctx"), get_int(query, "step"), query["fmt"][0])
        ret, content_type = json.dumps(render_src).encode(), "application/json"
      else:
        try: return self.stream_json(get_full_rewrite(trace.rewrites[i:=get_int(query, "ctx")][get_int(query, "idx")], i))
        except (KeyError, IndexError): status_code = 404
    elif url.path == "/ctxs": ret, content_type = json.dumps(ctxs).encode(), "application/json"
    elif url.path == "/get_profile" and profile_ret: ret, content_type = profile_ret, "application/octet-stream"
    else: status_code = 404

    # send response
    self.send_response(status_code)
    self.send_header('Content-Type', content_type)
    self.send_header('Content-Length', str(len(ret)))
    self.end_headers()
    return self.wfile.write(ret)

  def stream_json(self, source:Generator):
    try:
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-cache")
      self.end_headers()
      for r in source:
        self.wfile.write(f"data: {json.dumps(r)}\n\n".encode("utf-8"))
        self.wfile.flush()
      self.wfile.write("data: END\n\n".encode("utf-8"))
    # pass if client closed connection
    except (BrokenPipeError, ConnectionResetError): return

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

# NOTE: using HTTPServer forces a potentially slow socket.getfqdn
class TCPServerWithReuse(socketserver.TCPServer): allow_reuse_address = True

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

  ctxs = get_rewrites(trace:=load_pickle(args.kernels, default=RewriteTrace([], [], {})))
  profile_ret = get_profile(load_pickle(args.profile, default=[]))

  server = TCPServerWithReuse(('', PORT), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(f"*** started viz on {HOST}:{PORT}")
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"), flush=True)
  if len(getenv("BROWSER", "")) > 0: webbrowser.open(f"{HOST}:{PORT}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
