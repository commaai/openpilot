#!/usr/bin/env python3
import multiprocessing, pickle, functools, difflib, os, threading, json, time, sys, webbrowser, socket, argparse, decimal
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional
from tinygrad.helpers import colored, getenv, to_function_name, tqdm, unwrap, word_wrap
from tinygrad.ops import TrackedGraphRewrite, UOp, Ops, lines, GroupOp
from tinygrad.codegen.kernel import Kernel
from tinygrad.device import ProfileEvent, ProfileDeviceEvent, ProfileRangeEvent, ProfileGraphEvent

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.PRELOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.VCONST: "#e0e0e0",
               Ops.DEFINE_GLOBAL: "#ffe0b0", Ops.DEFINE_LOCAL: "#ffe0d0", Ops.DEFINE_ACC: "#f0ffe0", Ops.REDUCE_AXIS: "#FF6B6B",
               Ops.RANGE: "#c8a0e0", Ops.ASSIGN: "#e0ffc0", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#e8ffa0", Ops.WMMA: "#efefc0", Ops.VIEW: "#C8F9D4",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80",
               Ops.BLOCK: "#C4A484", Ops.BLOCKEND: "#C4A4A4", Ops.BUFFER: "#B0BDFF", Ops.COPY: "#a040a0"}

# ** API spec

@dataclass
class GraphRewriteMetadata:
  """Overview of a tracked rewrite to viz the sidebar"""
  loc: tuple[str, int]
  """File_path, Lineno"""
  code_line: str
  """The Python line calling graph_rewrite"""
  kernel_name: str
  """The kernel calling graph_rewrite"""
  upats: list[tuple[tuple[str, int], str, float]]
  """List of all the applied UPats"""

@dataclass
class GraphRewriteDetails(GraphRewriteMetadata):
  """Full details about a single call to graph_rewrite"""
  graphs: list[UOp]
  """Sink at every step of graph_rewrite"""
  diffs: list[list[str]]
  """.diff style before and after of the rewritten UOp child"""
  changed_nodes: list[list[int]]
  """Nodes that changed at every step of graph_rewrite"""
  kernel_code: Optional[str]
  """The program after all rewrites"""

# ** API functions

# NOTE: if any extra rendering in VIZ fails, we don't crash
def pcall(fxn:Callable[..., str], *args, **kwargs) -> str:
  try: return fxn(*args, **kwargs)
  except Exception as e: return f"ERROR: {e}"

def get_metadata(keys:list[Any], contexts:list[list[TrackedGraphRewrite]]) -> list[list[tuple[Any, TrackedGraphRewrite, GraphRewriteMetadata]]]:
  kernels: dict[str, list[tuple[Any, TrackedGraphRewrite, GraphRewriteMetadata]]] = {}
  for k,ctxs in tqdm(zip(keys, contexts), desc="preparing kernels"):
    name = to_function_name(k.name) if isinstance(k, Kernel) else str(k)
    for ctx in ctxs:
      if pickle.loads(ctx.sink).op is Ops.CONST: continue
      upats = [(upat.location, upat.printable(), tm) for _,_,upat,tm in ctx.matches if upat is not None]
      kernels.setdefault(name, []).append((k, ctx, GraphRewriteMetadata(ctx.loc, lines(ctx.loc[0])[ctx.loc[1]-1].strip(), name, upats)))
  return list(kernels.values())

def uop_to_json(x:UOp) -> dict[int, tuple[str, str, list[int], str, str]]:
  assert isinstance(x, UOp)
  graph: dict[int, tuple[str, str, list[int], str, str]] = {}
  excluded = set()
  for u in x.toposort:
    if u.op in {Ops.CONST, Ops.DEVICE}:
      excluded.add(u)
      continue
    argst = ("\n".join([f"{v.shape} / {v.strides}"+(f" / {v.offset}" if v.offset else "") for v in u.arg.views])) if u.op is Ops.VIEW else str(u.arg)
    label = f"{str(u.op).split('.')[1]}{(' '+word_wrap(argst.replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}"
    for idx,x in enumerate(u.src):
      if x.op is Ops.CONST: label += f"\nCONST{idx} {x.arg:g}"
      if x.op is Ops.DEVICE: label += f"\nDEVICE{idx} {x.arg}"
    graph[id(u)] = (label, str(u.dtype), [id(x) for x in u.src if x not in excluded], str(u.arg), uops_colors.get(u.op, "#ffffff"))
  return graph
def _replace_uop(base:UOp, replaces:dict[UOp, UOp]) -> UOp:
  if (found:=replaces.get(base)) is not None: return found
  ret = base.replace(src=tuple(_replace_uop(x, replaces) for x in base.src))
  if (final := replaces.get(ret)) is not None:
      return final
  replaces[base] = ret
  return ret
@functools.lru_cache(None)
def _prg(k:Kernel): return k.to_program().src
def get_details(k:Any, ctx:TrackedGraphRewrite, metadata:GraphRewriteMetadata) -> GraphRewriteDetails:
  g = GraphRewriteDetails(**asdict(metadata), graphs=[pickle.loads(ctx.sink)], diffs=[], changed_nodes=[],
                          kernel_code=pcall(_prg, k) if isinstance(k, Kernel) else None)
  replaces: dict[UOp, UOp] = {}
  sink = g.graphs[0]
  for i,(u0_b,u1_b,upat,_) in enumerate(ctx.matches):
    u0 = pickle.loads(u0_b)
    # if the match didn't result in a rewrite we move forward
    if u1_b is None:
      replaces[u0] = u0
      continue
    replaces[u0] = u1 = pickle.loads(u1_b)
    # first, rewrite this UOp with the current rewrite + all the matches in replaces
    new_sink = _replace_uop(sink, {**replaces})
    # sanity check
    if new_sink is sink: raise AssertionError(f"rewritten sink wasn't rewritten! {i} {unwrap(upat).location}")
    # update ret data
    g.changed_nodes.append([id(x) for x in u1.toposort if x.op is not Ops.CONST])
    g.diffs.append(list(difflib.unified_diff(pcall(str, u0).splitlines(), pcall(str, u1).splitlines())))
    g.graphs.append(sink:=new_sink)
  return g

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

    if (url:=urlparse(self.path)).path == "/":
      with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f: ret = f.read()
    elif (url:=urlparse(self.path)).path == "/profiler":
      with open(os.path.join(os.path.dirname(__file__), "perfetto.html"), "rb") as f: ret = f.read()
    elif self.path.startswith("/assets/") and '/..' not in self.path:
      try:
        with open(os.path.join(os.path.dirname(__file__), self.path.strip('/')), "rb") as f: ret = f.read()
        if url.path.endswith(".js"): content_type = "application/javascript"
        if url.path.endswith(".css"): content_type = "text/css"
      except FileNotFoundError: status_code = 404
    elif url.path == "/kernels":
      query = parse_qs(url.query)
      if (qkernel:=query.get("kernel")) is not None:
        g = get_details(*kernels[int(qkernel[0])][int(query["idx"][0])])
        jret: Any = {**asdict(g), "graphs": [uop_to_json(x) for x in g.graphs], "uops": [pcall(str,x) for x in g.graphs]}
      else: jret = [list(map(lambda x:asdict(x[2]), v)) for v in kernels]
      ret, content_type = json.dumps(jret).encode(), "application/json"
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

  kernels = get_metadata(*contexts) if contexts is not None else []

  if getenv("FUZZ_VIZ"):
    ret = [get_details(*args) for v in tqdm(kernels) for args in v]
    print(f"fuzzed {len(ret)} rewrite details")

  perfetto_profile = to_perfetto(profile) if profile is not None else None

  server = HTTPServer(('', PORT), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(f"*** started viz on {HOST}:{PORT}")
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"))
  if len(getenv("BROWSER", "")) > 0: webbrowser.open(f"{HOST}:{PORT}{'/profiler' if contexts is None else ''}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
