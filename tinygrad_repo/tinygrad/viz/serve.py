#!/usr/bin/env python3
import multiprocessing, pickle, functools, difflib, os, threading, json, time, sys, webbrowser, socket, argparse, decimal, socketserver
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from typing import Any, Callable, TypedDict, Generator
from tinygrad.helpers import colored, getenv, tqdm, unwrap, word_wrap, TRACEMETA
from tinygrad.ops import TrackedGraphRewrite, UOp, Ops, lines, GroupOp, srender, sint
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import ProfileEvent, ProfileDeviceEvent, ProfileRangeEvent, ProfileGraphEvent
from tinygrad.dtype import dtypes

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.VCONST: "#e0e0e0", Ops.REDUCE: "#FF5B5B",
               Ops.DEFINE_GLOBAL: "#ffe0b0", Ops.DEFINE_LOCAL: "#ffe0d0", Ops.DEFINE_ACC: "#f0ffe0", Ops.REDUCE_AXIS: "#FF6B6B",
               Ops.RANGE: "#c8a0e0", Ops.ASSIGN: "#909090", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#e8ffa0", Ops.WMMA: "#efefc0", Ops.VIEW: "#C8F9D4", Ops.MULTI: "#f6ccff", Ops.KERNEL: "#3e7f55", Ops.IGNORE: "#00C000",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80", Ops.BUFFER_VIEW: "#E5EAFF",
               Ops.BLOCK: "#C4A484", Ops.BLOCKEND: "#C4A4A4", Ops.BUFFER: "#B0BDFF", Ops.COPY: "#a040a0", Ops.FUSE: "#FFa500"}

# VIZ API

# NOTE: if any extra rendering in VIZ fails, we don't crash
def pcall(fxn:Callable[..., str], *args, **kwargs) -> str:
  err = kwargs.pop("err", "")
  try: return fxn(*args, **kwargs)
  except Exception: return f"ERROR in {fxn.__name__}\n{err}"

# ** Metadata for a track_rewrites scope

class GraphRewriteMetadata(TypedDict):
  loc: tuple[str, int]                   # [path, lineno] calling graph_rewrite
  match_count: int                       # total match count in this context
  code_line: str                         # source code calling graph_rewrite
  kernel_code: str|None                  # optionally render the final kernel code
  name: str|None                         # optional name of the rewrite
  depth: int                             # depth if it's a subrewrite

@functools.cache
def render_program(k:Kernel): return k.opts.render(k.uops)

def to_metadata(k:Any, v:TrackedGraphRewrite) -> GraphRewriteMetadata:
  return {"loc":v.loc, "match_count":len(v.matches), "name":v.name, "depth":v.depth, "code_line":lines(v.loc[0])[v.loc[1]-1].strip(),
          "kernel_code":pcall(render_program, k, err=f"ast = {k.ast}\nopts = {k.applied_opts}") if isinstance(k, Kernel) else None}

def get_metadata(keys:list[Any], contexts:list[list[TrackedGraphRewrite]]) -> list[tuple[str, list[GraphRewriteMetadata]]]:
  return [(k.name if isinstance(k, Kernel) else str(k), [to_metadata(k, v) for v in vals]) for k,vals in zip(keys, contexts)]

# ** Complete rewrite details for a graph_rewrite call

class GraphRewriteDetails(TypedDict):
  graph: dict                            # JSON serialized UOp for this rewrite step
  uop: str                               # strigified UOp for this rewrite step
  diff: list[str]|None                   # diff of the single UOp that changed
  changed_nodes: list[int]|None          # the changed UOp id + all its parents ids
  upat: tuple[tuple[str, int], str]|None # [loc, source_code] of the matched UPat

def shape_to_str(s:tuple[sint, ...]): return "(" + ','.join(srender(x) for x in s) + ")"
def mask_to_str(s:tuple[tuple[sint, sint], ...]): return "(" + ','.join(shape_to_str(x) for x in s) + ")"

def uop_to_json(x:UOp) -> dict[int, dict]:
  assert isinstance(x, UOp)
  graph: dict[int, dict] = {}
  excluded: set[UOp] = set()
  for u in (toposort:=x.toposort()):
    # always exclude DEVICE/CONST/UNIQUE
    if u.op in {Ops.DEVICE, Ops.CONST, Ops.UNIQUE}: excluded.add(u)
    # only exclude CONST VIEW source if it has no other children in the graph
    if u.op is Ops.CONST and len(u.src) != 0 and all(cr.op is Ops.CONST for c in u.src[0].children if (cr:=c()) is not None and cr in toposort):
      excluded.update(u.src)
  for u in toposort:
    if u in excluded: continue
    argst = str(u.arg)
    if u.op is Ops.VIEW:
      argst = ("\n".join([f"{shape_to_str(v.shape)} / {shape_to_str(v.strides)}"+("" if v.offset == 0 else f" / {srender(v.offset)}")+
                          (f"\nMASK {mask_to_str(v.mask)}" if v.mask is not None else "") for v in unwrap(u.st).views]))
    label = f"{str(u.op).split('.')[1]}{(chr(10)+word_wrap(argst.replace(':', ''))) if u.arg is not None else ''}"
    if u.dtype != dtypes.void: label += f"\n{u.dtype}"
    for idx,x in enumerate(u.src):
      if x in excluded:
        if x.op is Ops.CONST and dtypes.is_float(u.dtype): label += f"\nCONST{idx} {x.arg:g}"
        else: label += f"\n{x.op.name}{idx} {x.arg}"
    try:
      if u.op not in {Ops.VIEW, Ops.BUFFER, Ops.KERNEL, Ops.ASSIGN, Ops.COPY, Ops.SINK, *GroupOp.Buffer} and u.st is not None:
        label += f"\n{shape_to_str(u.shape)}"
    except Exception:
      label += "\n<ISSUE GETTING SHAPE>"
    # NOTE: kernel already has metadata in arg
    if TRACEMETA >= 2 and u.metadata is not None and u.op is not Ops.KERNEL: label += "\n"+repr(u.metadata)
    graph[id(u)] = {"label":label, "src":[id(x) for x in u.src if x not in excluded], "color":uops_colors.get(u.op, "#ffffff")}
  return graph

def get_details(k:Any, ctx:TrackedGraphRewrite) -> Generator[GraphRewriteDetails, None, None]:
  yield {"graph":uop_to_json(next_sink:=ctx.sink), "uop":str(ctx.sink), "changed_nodes":None, "diff":None, "upat":None}
  replaces: dict[UOp, UOp] = {}
  for u0,u1,upat in tqdm(ctx.matches):
    replaces[u0] = u1
    try: new_sink = next_sink.substitute(replaces)
    except RecursionError as e: new_sink = UOp(Ops.NOOP, arg=str(e))
    yield {"graph":(sink_json:=uop_to_json(new_sink)), "uop":str(new_sink), "changed_nodes":[id(x) for x in u1.toposort() if id(x) in sink_json],
           "diff":list(difflib.unified_diff(pcall(str, u0).splitlines(), pcall(str, u1).splitlines())), "upat":(upat.location, upat.printable())}
    if not ctx.bottom_up: next_sink = new_sink

# Profiler API
devices:dict[str, tuple[decimal.Decimal, decimal.Decimal, int]] = {}
def prep_ts(device:str, ts:decimal.Decimal, is_copy): return int(decimal.Decimal(ts) + devices[device][is_copy])
def dev_to_pid(device:str, is_copy=False): return {"pid": devices[device][2], "tid": int(is_copy)}
def dev_ev_to_perfetto_json(ev:ProfileDeviceEvent):
  devices[ev.device] = (ev.comp_tdiff, ev.copy_tdiff if ev.copy_tdiff is not None else ev.comp_tdiff, len(devices))
  return [{"name": "process_name", "ph": "M", "pid": dev_to_pid(ev.device)['pid'], "args": {"name": ev.device}},
          {"name": "thread_name", "ph": "M", "pid": dev_to_pid(ev.device)['pid'], "tid": 0, "args": {"name": "COMPUTE"}},
          {"name": "thread_name", "ph": "M", "pid": dev_to_pid(ev.device)['pid'], "tid": 1, "args": {"name": "COPY"}}]
def range_ev_to_perfetto_json(ev:ProfileRangeEvent):
  return [{"name": ev.name, "ph": "X", "ts": prep_ts(ev.device, ev.st, ev.is_copy), "dur": float(ev.en-ev.st), **dev_to_pid(ev.device, ev.is_copy)}]
def graph_ev_to_perfetto_json(ev:ProfileGraphEvent, reccnt):
  ret = []
  for i,e in enumerate(ev.ents):
    st, en = ev.sigs[e.st_id], ev.sigs[e.en_id]
    ret += [{"name": e.name, "ph": "X", "ts": prep_ts(e.device, st, e.is_copy), "dur": float(en-st), **dev_to_pid(e.device, e.is_copy)}]
    for dep in ev.deps[i]:
      d = ev.ents[dep]
      ret += [{"ph": "s", **dev_to_pid(d.device, d.is_copy), "id": reccnt+len(ret), "ts": prep_ts(d.device, ev.sigs[d.en_id], d.is_copy), "bp": "e"}]
      ret += [{"ph": "f", **dev_to_pid(e.device, e.is_copy), "id": reccnt+len(ret)-1, "ts": prep_ts(e.device, st, e.is_copy), "bp": "e"}]
  return ret
def to_perfetto(profile:list[ProfileEvent]):
  # Start json with devices.
  prof_json = [x for ev in profile if isinstance(ev, ProfileDeviceEvent) for x in dev_ev_to_perfetto_json(ev)]
  for ev in tqdm(profile, desc="preparing profile"):
    if isinstance(ev, ProfileRangeEvent): prof_json += range_ev_to_perfetto_json(ev)
    elif isinstance(ev, ProfileGraphEvent): prof_json += graph_ev_to_perfetto_json(ev, reccnt=len(prof_json))
  return json.dumps({"traceEvents": prof_json}).encode() if len(prof_json) > 0 else None

# ** HTTP server

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    ret, status_code, content_type = b"", 200, "text/html"

    if (fn:={"/":"index", "/profiler":"perfetto", "/prof":"profiler"}.get((url:=urlparse(self.path)).path)):
      with open(os.path.join(os.path.dirname(__file__), f"{fn}.html"), "rb") as f: ret = f.read()
    elif self.path.startswith(("/assets/", "/js/")) and '/..' not in self.path:
      try:
        with open(os.path.join(os.path.dirname(__file__), self.path.strip('/')), "rb") as f: ret = f.read()
        if url.path.endswith(".js"): content_type = "application/javascript"
        if url.path.endswith(".css"): content_type = "text/css"
      except FileNotFoundError: status_code = 404
    elif url.path == "/kernels":
      if "kernel" in (query:=parse_qs(url.query)):
        kidx, ridx = int(query["kernel"][0]), int(query["idx"][0])
        try:
          # stream details
          self.send_response(200)
          self.send_header("Content-Type", "text/event-stream")
          self.send_header("Cache-Control", "no-cache")
          self.end_headers()
          for r in get_details(contexts[0][kidx], contexts[1][kidx][ridx]):
            self.wfile.write(f"data: {json.dumps(r)}\n\n".encode("utf-8"))
            self.wfile.flush()
          self.wfile.write("data: END\n\n".encode("utf-8"))
          return self.wfile.flush()
        # pass if client closed connection
        except (BrokenPipeError, ConnectionResetError): return
      ret, content_type = json.dumps(kernels).encode(), "application/json"
    elif url.path == "/get_profile" and perfetto_profile is not None: ret, content_type = perfetto_profile, "application/json"
    else: status_code = 404

    # send response
    self.send_response(status_code)
    self.send_header('Content-Type', content_type)
    self.send_header('Content-Length', str(len(ret)))
    self.end_headers()
    return self.wfile.write(ret)

# ** main loop

def reloader():
  mtime = os.stat(__file__).st_mtime
  while not stop_reloader.is_set():
    if mtime != os.stat(__file__).st_mtime:
      print("reloading server...")
      os.execv(sys.executable, [sys.executable] + sys.argv)
    time.sleep(0.1)

def load_pickle(path:str):
  if path is None or not os.path.exists(path): return None
  with open(path, "rb") as f: return pickle.load(f)

# NOTE: using HTTPServer forces a potentially slow socket.getfqdn
class TCPServerWithReuse(socketserver.TCPServer): allow_reuse_address = True

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--kernels', type=str, help='Path to kernels', default=None)
  parser.add_argument('--profile', type=str, help='Path profile', default=None)
  args = parser.parse_args()

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    if s.connect_ex(((HOST:="http://127.0.0.1").replace("http://", ""), PORT:=getenv("PORT", 8000))) == 0:
      raise RuntimeError(f"{HOST}:{PORT} is occupied! use PORT= to change.")
  stop_reloader = threading.Event()
  multiprocessing.current_process().name = "VizProcess"    # disallow opening of devices
  st = time.perf_counter()
  print("*** viz is starting")

  contexts, profile = load_pickle(args.kernels), load_pickle(args.profile)

  # NOTE: this context is a tuple of list[keys] and list[values]
  kernels = get_metadata(*contexts) if contexts is not None else []

  perfetto_profile = to_perfetto(profile) if profile is not None else None

  server = TCPServerWithReuse(('', PORT), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(f"*** started viz on {HOST}:{PORT}")
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"))
  if len(getenv("BROWSER", "")) > 0: webbrowser.open(f"{HOST}:{PORT}{'/profiler' if contexts is None else ''}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
