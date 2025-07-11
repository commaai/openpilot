#!/usr/bin/env python3
import multiprocessing, pickle, difflib, os, threading, json, time, sys, webbrowser, socket, argparse, socketserver, functools, decimal, codecs
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from typing import Any, TypedDict, Generator
from tinygrad.helpers import colored, getenv, tqdm, unwrap, word_wrap, TRACEMETA
from tinygrad.uop.ops import TrackedGraphRewrite, TracingKey, UOp, Ops, printable, GroupOp, srender, sint
from tinygrad.device import ProfileEvent, ProfileDeviceEvent, ProfileRangeEvent, ProfileGraphEvent, ProfileGraphEntry, ProfilePointEvent
from tinygrad.dtype import dtypes

uops_colors = {Ops.LOAD: "#ffc0c0", Ops.STORE: "#87CEEB", Ops.CONST: "#e0e0e0", Ops.VCONST: "#e0e0e0", Ops.REDUCE: "#FF5B5B",
               Ops.DEFINE_GLOBAL: "#ffe0b0", Ops.DEFINE_LOCAL: "#ffe0d0", Ops.DEFINE_REG: "#f0ffe0", Ops.REDUCE_AXIS: "#FF6B6B",
               Ops.RANGE: "#c8a0e0", Ops.ASSIGN: "#909090", Ops.BARRIER: "#ff8080", Ops.IF: "#c8b0c0", Ops.SPECIAL: "#c0c0ff",
               Ops.INDEX: "#e8ffa0", Ops.WMMA: "#efefc0", Ops.VIEW: "#C8F9D4", Ops.MULTI: "#f6ccff", Ops.KERNEL: "#3e7f55",
               **{x:"#D8F9E4" for x in GroupOp.Movement}, **{x:"#ffffc0" for x in GroupOp.ALU}, Ops.THREEFRY:"#ffff80", Ops.BUFFER_VIEW: "#E5EAFF",
               Ops.BLOCK: "#C4A484", Ops.BLOCKEND: "#C4A4A4", Ops.BUFFER: "#B0BDFF", Ops.COPY: "#a040a0", Ops.FUSE: "#FFa500",
               Ops.ALLREDUCE: "#ff40a0", Ops.GBARRIER: "#FFC14D", Ops.MSELECT: "#d040a0", Ops.MSTACK: "#d040a0"}

# VIZ API

# ** Metadata for a track_rewrites scope

ref_map:dict[Any, int] = {}
def get_metadata(keys:list[TracingKey], contexts:list[list[TrackedGraphRewrite]]) -> list[dict]:
  ret = []
  for i,(k,v) in enumerate(zip(keys, contexts)):
    steps = [{"name":s.name, "loc":s.loc, "depth":s.depth, "match_count":len(s.matches), "code_line":printable(s.loc)} for s in v]
    ret.append({"name":k.display_name, "fmt":k.fmt, "steps":steps})
    for key in k.keys: ref_map[key] = i
  return ret

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
    argst = codecs.decode(str(u.arg), "unicode_escape")
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
    if (ref:=ref_map.get(u.arg.ast) if u.op is Ops.KERNEL else None) is not None: label += f"\ncodegen@{ctxs[ref]['name']}"
    # NOTE: kernel already has metadata in arg
    if TRACEMETA >= 2 and u.metadata is not None and u.op is not Ops.KERNEL: label += "\n"+repr(u.metadata)
    graph[id(u)] = {"label":label, "src":[id(x) for x in u.src if x not in excluded], "color":uops_colors.get(u.op, "#ffffff"),
                    "ref":ref, "tag":u.tag}
  return graph

@functools.cache
def _reconstruct(a:int):
  op, dtype, src, arg, tag = contexts[2][a]
  arg = type(arg)(_reconstruct(arg.ast), arg.metadata) if op is Ops.KERNEL else arg
  return UOp(op, dtype, tuple(_reconstruct(s) for s in src), arg, tag)

def get_details(ctx:TrackedGraphRewrite) -> Generator[GraphRewriteDetails, None, None]:
  yield {"graph":uop_to_json(next_sink:=_reconstruct(ctx.sink)), "uop":str(next_sink), "changed_nodes":None, "diff":None, "upat":None}
  replaces: dict[UOp, UOp] = {}
  for u0_num,u1_num,upat_loc in tqdm(ctx.matches):
    replaces[u0:=_reconstruct(u0_num)] = u1 = _reconstruct(u1_num)
    try: new_sink = next_sink.substitute(replaces)
    except RuntimeError as e: new_sink = UOp(Ops.NOOP, arg=str(e))
    yield {"graph":(sink_json:=uop_to_json(new_sink)), "uop":str(new_sink), "changed_nodes":[id(x) for x in u1.toposort() if id(x) in sink_json],
           "diff":list(difflib.unified_diff(str(u0).splitlines(), str(u1).splitlines())), "upat":(upat_loc, printable(upat_loc))}
    if not ctx.bottom_up: next_sink = new_sink

# Profiler API

DevEvent = ProfileRangeEvent|ProfileGraphEntry|ProfilePointEvent
def flatten_events(profile:list[ProfileEvent]) -> Generator[tuple[decimal.Decimal, decimal.Decimal|None, DevEvent], None, None]:
  for e in profile:
    if isinstance(e, ProfileRangeEvent): yield (e.st, e.en, e)
    if isinstance(e, ProfilePointEvent): yield (e.st, None, e)
    if isinstance(e, ProfileGraphEvent):
      for ent in e.ents: yield (e.sigs[ent.st_id], e.sigs[ent.en_id], ent)

# timeline layout stacks events in a contiguous block. When a late starter finishes late, there is whitespace in the higher levels.
def timeline_layout(events:list[tuple[int, int, float, DevEvent]]) -> dict:
  shapes:list[dict] = []
  levels:list[int] = []
  for st,et,dur,e in events:
    if dur == 0: continue
    # find a free level to put the event
    depth = next((i for i,level_et in enumerate(levels) if st>=level_et), len(levels))
    if depth < len(levels): levels[depth] = et
    else: levels.append(et)
    name, cat = e.name, None
    if (ref:=ref_map.get(name)) is not None: name = ctxs[ref]["name"]
    elif isinstance(e.name, TracingKey):
      name, cat = e.name.display_name, e.name.cat
      ref = next((v for k in e.name.keys if (v:=ref_map.get(k)) is not None), None)
    shapes.append({"name":name, "ref":ref, "st":st, "dur":dur, "depth":depth, "cat":cat})
  return {"shapes":shapes, "maxDepth":len(levels)}

def mem_layout(events:list[tuple[int, int, float, DevEvent]]) -> dict:
  step, peak, mem = 0, 0, 0
  shps:dict[int, dict] = {}
  temp:dict[int, dict] = {}
  timestamps:list[int] = []
  for st,_,_,e in events:
    if not isinstance(e, ProfilePointEvent): continue
    if e.name == "alloc":
      shps[e.ref] = temp[e.ref] = {"x":[step], "y":[mem], "arg":e.arg}
      timestamps.append(int(e.st))
      step += 1
      mem += e.arg["nbytes"]
      if mem > peak: peak = mem
    if e.name == "free":
      timestamps.append(int(e.st))
      step += 1
      mem -= (removed:=temp.pop(e.ref))["arg"]["nbytes"]
      removed["x"].append(step)
      removed["y"].append(removed["y"][-1])
      for k,v in temp.items():
        if k > e.ref:
          v["x"] += [step, step]
          v["y"] += [v["y"][-1], v["y"][-1]-removed["arg"]["nbytes"]]
  for v in temp.values():
    v["x"].append(step)
    v["y"].append(v["y"][-1])
  return {"shapes":list(shps.values()), "peak":peak, "timestamps":timestamps}

def get_profile(profile:list[ProfileEvent]):
  # start by getting the time diffs
  devs = {e.device:(e.comp_tdiff, e.copy_tdiff if e.copy_tdiff is not None else e.comp_tdiff) for e in profile if isinstance(e,ProfileDeviceEvent)}
  # map events per device
  dev_events:dict[str, list[tuple[int, int, float, DevEvent]]] = {}
  min_ts:int|None = None
  max_ts:int|None = None
  for ts,en,e in flatten_events(profile):
    time_diff = devs[e.device][e.__dict__.get("is_copy",False)] if e.device in devs else decimal.Decimal(0)
    # ProfilePointEvent records perf_counter, offset other events by GPU time diff
    st = int(ts) if isinstance(e, ProfilePointEvent) else int(ts+time_diff)
    et = st if en is None else int(en+time_diff)
    dev_events.setdefault(e.device,[]).append((st, et, 0. if en is None else float(en-ts), e))
    if min_ts is None or st < min_ts: min_ts = st
    if max_ts is None or et > max_ts: max_ts = et
  # return layout of per device events
  for events in dev_events.values(): events.sort(key=lambda v:v[0])
  dev_layout = {k:{"timeline":timeline_layout(v), "mem":mem_layout(v)} for k,v in dev_events.items()}
  return json.dumps({"layout":dev_layout, "st":min_ts, "et":max_ts}).encode("utf-8")

# ** HTTP server

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
    elif url.path == "/ctxs":
      if "ctx" in (query:=parse_qs(url.query)):
        kidx, ridx = int(query["ctx"][0]), int(query["idx"][0])
        try:
          # stream details
          self.send_response(200)
          self.send_header("Content-Type", "text/event-stream")
          self.send_header("Cache-Control", "no-cache")
          self.end_headers()
          for r in get_details(contexts[1][kidx][ridx]):
            self.wfile.write(f"data: {json.dumps(r)}\n\n".encode("utf-8"))
            self.wfile.flush()
          self.wfile.write("data: END\n\n".encode("utf-8"))
          return self.wfile.flush()
        # pass if client closed connection
        except (BrokenPipeError, ConnectionResetError): return
      ret, content_type = json.dumps(ctxs).encode(), "application/json"
    elif url.path == "/get_profile" and profile_ret is not None: ret, content_type = profile_ret, "application/json"
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
  ctxs = get_metadata(*contexts[:2]) if contexts is not None else []

  profile_ret = get_profile(profile) if profile is not None else None

  server = TCPServerWithReuse(('', PORT), Handler)
  reloader_thread = threading.Thread(target=reloader)
  reloader_thread.start()
  print(f"*** started viz on {HOST}:{PORT}")
  print(colored(f"*** ready in {(time.perf_counter()-st)*1e3:4.2f}ms", "green"), flush=True)
  if len(getenv("BROWSER", "")) > 0: webbrowser.open(f"{HOST}:{PORT}{'/profiler' if contexts is None else ''}")
  try: server.serve_forever()
  except KeyboardInterrupt:
    print("*** viz is shutting down...")
    stop_reloader.set()
