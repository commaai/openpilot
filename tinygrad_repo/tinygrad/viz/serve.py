#!/usr/bin/env python3
import multiprocessing, pickle, functools, difflib, os, threading, json, time, sys, webbrowser, socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Optional
from tinygrad.helpers import colored, getenv, to_function_name, tqdm, unwrap, word_wrap
from tinygrad.ops import TrackedRewriteContext, UOp, Ops, lines, GroupOp
from tinygrad.codegen.kernel import Kernel

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.PRELOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.VCONST: "#e0e0e0",
               Ops.DEFINE_GLOBAL: "#ffe0b0", Ops.DEFINE_LOCAL: "#ffe0d0", Ops.DEFINE_ACC: "#f0ffe0", Ops.REDUCE_AXIS: "#FF6B6B",
               Ops.RANGE: "#c8a0e0", Ops.ASSIGN: "#e0ffc0", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#e8ffa0", Ops.WMMA: "#efefc0", Ops.VIEW: "#C8F9D4", **{x:"#ffffc0" for x in GroupOp.ALU},
               Ops.BLOCK: "#C4A484", Ops.BLOCKEND: "#C4A4A4", Ops.BUFFER: "#B0BDFF",}

# ** API spec

@dataclass
class GraphRewriteMetadata:
  """Specifies metadata about a single call to graph_rewrite"""
  loc: Tuple[str, int]
  """File_path, Lineno"""
  code_line: str
  """The Python line calling graph_rewrite"""
  kernel_name: Optional[str]
  """The kernel calling graph_rewrite"""
  upats: List[Tuple[Tuple[str, int], str, float]]
  """List of all the applied UPats"""

@dataclass
class GraphRewriteDetails(GraphRewriteMetadata):
  """Full details about a single call to graph_rewrite"""
  graphs: List[UOp]
  """Sink at every step of graph_rewrite"""
  diffs: List[List[str]]
  """.diff style before and after of the rewritten UOp child"""
  changed_nodes: List[List[int]]
  """Nodes that changed at every step of graph_rewrite"""
  kernel_code: Optional[str]
  """The program after all rewrites"""

# ** API functions

def pcall(fxn, *args, **kwargs):
  try: return fxn(*args, **kwargs)
  except Exception as e: return f"ERROR: {e}"

def get_metadata(contexts:List[Tuple[Any, List[TrackedRewriteContext]]]) -> List[List[Tuple[Any, TrackedRewriteContext, GraphRewriteMetadata]]]:
  kernels: Dict[Optional[str], List[Tuple[Any, TrackedRewriteContext, GraphRewriteMetadata]]] = {}
  for k,ctxs in contexts:
    name = to_function_name(k.name) if isinstance(k, Kernel) else k
    for ctx in ctxs:
      if ctx.sink.op is Ops.CONST: continue
      upats = [(upat.location, upat.printable(), tm) for _,_,upat,tm in ctx.matches if upat is not None]
      if name not in kernels: kernels[name] = []
      kernels[name].append((k, ctx, GraphRewriteMetadata(ctx.loc, lines(ctx.loc[0])[ctx.loc[1]-1].strip(), name, upats)))
  return list(kernels.values())

def uop_to_json(x:UOp) -> Dict[int, Tuple[str, str, List[int], str, str]]:
  assert isinstance(x, UOp)
  graph: Dict[int, Tuple[str, str, List[int], str, str]] = {}
  for u in x.toposort:
    if u.op is Ops.CONST: continue
    argst = ("\n".join([f"{v.shape} / {v.strides}"+(f" / {v.offset}" if v.offset else "") for v in u.arg.views])) if u.op is Ops.VIEW else str(u.arg)
    label = f"{str(u.op).split('.')[1]}{(' '+word_wrap(argst.replace(':', ''))) if u.arg is not None else ''}\n{str(u.dtype)}"
    for idx,x in enumerate(u.src):
      if x.op is Ops.CONST: label += f"\nCONST{idx} {x.arg:g}"
    graph[id(u)] = (label, str(u.dtype), [id(x) for x in u.src if x.op is not Ops.CONST], str(u.arg), uops_colors.get(u.op, "#ffffff"))
  return graph
def _replace_uop(base:UOp, replaces:Dict[UOp, UOp]) -> UOp:
  if (found:=replaces.get(base)) is not None: return found
  ret = base.replace(src=tuple(_replace_uop(x, replaces) for x in base.src))
  if (final := replaces.get(ret)) is not None:
      return final
  replaces[base] = ret
  return ret
@functools.lru_cache(None)
def _prg(k:Optional[Kernel]) -> Optional[str]:
  try: return k.to_program().src if isinstance(k, Kernel) else None
  except Exception: return None
def get_details(k:Any, ctx:TrackedRewriteContext, metadata:GraphRewriteMetadata) -> GraphRewriteDetails:
  g = GraphRewriteDetails(**asdict(metadata), graphs=[ctx.sink], diffs=[], changed_nodes=[], kernel_code=pcall(_prg, k))
  replaces: Dict[UOp, UOp] = {}
  sink = ctx.sink
  for i,(u0,u1,upat,_) in enumerate(ctx.matches):
    if ctx.bottom_up: replaces = {} # if it's bottom_up it's single pass
    replaces[u0] = u0 if u1 is None else u1
    # if the match didn't result in a rewrite we move forward
    if u1 is None: continue
    # first, rewrite this UOp with the current rewrite + all the matches in replaces
    new_sink = _replace_uop(sink, {**replaces})
    # sanity check
    if new_sink is sink: raise AssertionError(f"rewritten sink wasn't rewritten! {i} {unwrap(upat).location}")
    # update ret data
    g.changed_nodes.append([id(x) for x in u1.toposort if x.op is not Ops.CONST])
    g.diffs.append(list(difflib.unified_diff(pcall(str, u0).splitlines(), pcall(str, u1).splitlines())))
    g.graphs.append(sink:=new_sink)
  return g

# ** HTTP server

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    ret, status_code, content_type = b"", 200, "text/html"

    if (url:=urlparse(self.path)).path == "/":
      with open(os.path.join(os.path.dirname(__file__), "index.html"), "rb") as f: ret = f.read()
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

if __name__ == "__main__":
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    if s.connect_ex(((HOST:="http://127.0.0.1").replace("http://", ""), PORT:=getenv("PORT", 8000))) == 0:
      raise RuntimeError(f"{HOST}:{PORT} is occupied! use PORT= to change.")
  stop_reloader = threading.Event()
  multiprocessing.current_process().name = "VizProcess"    # disallow opening of devices
  st = time.perf_counter()
  print("*** viz is starting")
  with open(sys.argv[1], "rb") as f: contexts: List[Tuple[Any, List[TrackedRewriteContext]]] = pickle.load(f)
  print("*** unpickled saved rewrites")
  kernels = get_metadata(contexts)
  if getenv("FUZZ_VIZ"):
    ret = [get_details(*args) for v in tqdm(kernels) for args in v]
    print(f"fuzzed {len(ret)} rewrite details")
  print("*** loaded kernels")
  server = HTTPServer(('', PORT), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(f"*** started viz on {HOST}:{PORT}")
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"))
  if getenv("BROWSER", 0): webbrowser.open(f"{HOST}:{PORT}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
