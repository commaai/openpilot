#!/usr/bin/env python3
import argparse, pathlib, signal, struct, json, os, itertools, heapq
os.environ["VIZ"] = "0"
if hasattr(signal, "SIGPIPE"): signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from typing import Iterator
from tinygrad.viz import serve as viz
from tinygrad.viz.serve import fmt_colored
from tinygrad.uop.ops import RewriteTrace
from tinygrad.helpers import temp, ansistrip, colored, time_to_str, ansilen, ProfilePointEvent, ProfileRangeEvent, TracingKey, unwrap, NO_COLOR, DEBUG

# profile decoder used in CLI and tests
def decode_profile(data:bytes) -> dict:
  ret, off = data, 0
  def u(fmt:str) -> tuple:
    nonlocal off
    vals = struct.unpack_from(fmt, ret, off)
    off += struct.calcsize(fmt)
    return vals
  total_dur, global_peak, index_len, layout_len = u("<IQII")
  strings, dtypes, markers = json.loads(ret[off:off+index_len]).values()
  off += index_len
  layout:dict[str, dict] = {}
  # 0 means None, otherwise it's an enum value
  def option(i:int) -> int|None: return None if i == 0 else i-1
  for _ in range(layout_len):
    klen = u("<B")[0]
    k = ret[off:off+klen].decode()
    off += klen
    event_type, event_count = u("<BI")
    layout[k] = v = {"event_type":event_type, "events":[]}
    if event_type == 0:
      for _ in range(event_count):
        name, ref, key, st, dur, fmt = u("<IIIIfI")
        v["events"].append({"name":strings[name], "ref":option(ref), "key":option(key), "st":st, "dur":dur, "fmt":json.loads(strings[fmt])})
    else:
      v["linear"] = u("<B")[0]
      v["peak"] = u("<Q")[0]
      for _ in range(event_count):
        if v["linear"]:
          ts, value = u("<IQ")
          v["events"].append({"event":"freq", "ts":ts, "value":value})
        else:
          alloc, ts, key = u("<BII")
          if alloc: v["events"].append({"event":"alloc", "ts":ts, "key":key, "arg": {"dtype":strings[u("<I")[0]], "sz":u("<Q")[0]}})
          else: v["events"].append({"event":"free", "ts":ts, "key":key, "arg": {"users":[(k, strings[rep], num, mode) \
              for k,rep,num,mode in [u("<IIIB") for _ in range(u("<I")[0])]]}})
  return {"dur":total_dur, "peak":global_peak, "layout":layout, "markers":markers}

def to_str(k:str, v) -> str:
  if k == "FLOPS" or k.startswith("B/s"): return f"{v*1e-9:.0f} G{k}" if v < 1e13 else f"{v*1e-12:.0f} T{k}"
  if k == "B": return next((f"{v/s:.0f} {u}" for s,u in ((1e9,"GB"),(1e6,"MB"),(1e3,"KB")) if v>=s), f"{v:.0f} B")
  return f"{k}={v}"
def fmt_data(data:dict) -> str: return "  ".join((p:=to_str(k, v))+" "*max(0, 14-ansilen(p)) for k,v in data.items())

def get(data:dict, key:str):
  for k,v in data.items():
    if ansistrip(k) == key: return v
  import difflib
  match = difflib.get_close_matches(key, [ansistrip(k) for k in data], n=1, cutoff=0.6)
  raise RuntimeError(f'item "{key}" not found in list'+(f", did you mean {match[0]!r}?" if match else ''))

def main(args) -> None:
  viz.load_rewrites(viz_data:=viz.VizData(viz.load_pickle(args.rewrites_path, default=RewriteTrace([], [], {}))))

  def emit(val, to_str=str) -> str: return json.dumps(val if isinstance(val, dict) else {"value":val}) if args.json else to_str(val)

  def print_step(step:dict, print_graph=False, reconstruct_matches=False) -> None:
    data = viz.get_render(viz_data, step["query"])
    if isinstance(data.get("value"), Iterator):
      for m in data["value"]:
        if print_graph and "graph" in m and not args.json:
          for k,v in m["graph"].items():
            print(f"[{k}] {' '.join((lines:=v['label'].splitlines())[:5])}{'...' if len(lines) > 5 else ''}"+(f" tag={v['tag']}" if v['tag'] else ''))
            if v["src"]:
              print("  src: "+", ".join([f"{i}->[{x}]" for i,x in v["src"][:5]])+(f", ... and {len(v['src'])-5} more" if len(v["src"]) > 5 else ""))
        elif "uop" in m: print(emit(m["graph"] if print_graph else m["uop"]))
        if not reconstruct_matches: return None
        if m.get("diff"):
          loc = pathlib.Path(m["upat"][0][0])
          print(emit(f"{loc.parent.name}/{loc.name}:{m['upat'][0][1]}\n{m['upat'][1]}"))
          for line in m["diff"]: print(emit(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None)))
    if data.get("src") is not None: print(emit(data["src"]))

  profile_bytes = viz.get_profile(viz_data, viz.load_pickle(args.profile_path, default=[]))
  if profile_bytes is None: raise RuntimeError(f"empty profile in {args.profile_path}")
  profile = decode_profile(profile_bytes)
  profile["layout"].update([(f'{c["name"][5:]}{" SQTT" if s["name"].endswith("PKTS") else ""} {s["name"]}', s["data"]) for c in viz_data.ctxs
                            if c["name"].startswith("SQTT") for s in c["steps"] if s["name"].endswith(("PMC", "PKTS"))])
  if args.list and not args.src: return print("\n".join(emit(fmt_colored(k)) for k in ["ALL"]+list(profile["layout"])))

  # ** SQTT printer
  data = None if not args.src else get(profile["layout"], args.src[0])
  if args.src and "SQTT" in args.src[0]:
    # modern terminals support 24-bit color
    def hex_colored(st:str, color:str) -> str: return f"\x1b[38;2;{int(color[1:3],16)};{int(color[3:5],16)};{int(color[5:7],16)}m{st}\x1b[0m"
    print(emit(f"{'Clk':<12} {'Unit':<20} {'Op':<22} {'Dur':<4} {'Delay':<4} {'Info'}"))
    print(emit("-" * 100))
    pc_map:dict[int, str] = {}
    pkt_idxs:dict[str, itertools.count] = {}
    dispatch_to_inst:dict[str, tuple[str, int]] = {}
    inst_st:int|None = None
    for e in viz.sqtt_timeline(*unwrap(data)):
      if isinstance(e, ProfilePointEvent) and e.key == 'pcMap': pc_map = e.arg
      if not isinstance(e, ProfileRangeEvent): continue
      if inst_st is None: inst_st = int(e.st)
      assert isinstance(e.name, TracingKey)
      op_name, ret, info = e.name.display_name, json.loads(e.name.ret[4:]) if e.name.ret else {}, ""
      color = next((v for k,v in viz.wave_colors.items() if k in op_name), None)
      op_str = hex_colored(op_name, color) if color and not NO_COLOR else op_name
      inst, phase, delay = None, None, 0
      idx = next(pkt_idxs.setdefault(e.device, itertools.count()))
      if e.device.startswith("WAVE"):
        inst = f"0x{pc:05x} {pc_map[pc]}" if (pc:=ret.get("pc")) is not None else f"{'':7} {op_name}"
        dispatch_to_inst[f"{e.device}-{idx}"] = (inst, int(e.st))
        phase = "DISPATCH"
      if (link:=ret.get("link")) is not None:
        inst, dispatch_st = dispatch_to_inst[link]
        phase, delay = "EXEC", int(e.st) - dispatch_st
      if inst and phase: info = f"{phase:<8} {inst}"
      unit = e.device.replace(" ", "-")
      row = {"clk":int(e.st)-inst_st, "cycle":int(e.st), "unit":unit, "op":op_name, "dur":int(unwrap(e.en)-e.st), "delay":delay or "", "info":info}
      print(emit(row, lambda _: f"{row['clk']:<12} {unit:<20} {op_str}{' '*(22-ansilen(op_str))} {row['dur']:<4} {str(row['delay']):<4} {info}"))

  # ** PMC printer
  elif args.src and "PMC" in args.src[0]:
    pmc = viz.unpack_pmc(unwrap(data))
    pmc_fmt:list[str] = []
    for name,val,*detail in pmc["rows"]:
      pmc_fmt += [f"{name} {val}"]+([" ".join(f"{k}={v}" for k,v in zip(detail[0]["cols"], r)) for r in detail[0]["rows"]] if detail else [])
    print(emit(pmc, lambda _: "\n".join(pmc_fmt)))

  # ** Memory printer
  elif data is not None and data["event_type"] == 1:
    print(emit({"peak":data["peak"]}, lambda _: f"Peak: {data['peak']}"+"\n"+f"{'TS':<10}  {'Event':<6}  {'Key':>8}  Info"))
    for e in data["events"]:
      info = str(arg:=e.pop("arg", {}))
      if e["event"] == "free": info = ', '.join([f"{fmt_colored(k)} {['read','write','write+read'][m]}@data{n}" for _,k,n,m in arg["users"]])
      print(emit({**e, "info":info}, lambda _: f"{e['ts']:<10}  {e['event']:<6}  {e.get('key', ''):>8}  {info}"))

  # ** Profiler printer
  else:
    timelines = [(n,l) for n,l in profile["layout"].items() if isinstance(l, dict) and l.get("event_type") == 0]
    def produce_top_kernels() -> Iterator[dict]:
      tagged = ((n,e) for n,l in timelines for e in l["events"]) if not args.src else ((args.src[0],e) for e in unwrap(data)["events"])
      agg:dict[tuple[str,str], tuple[float, int, int|None, dict[str, float]]] = {} # map (device, kernel name) to (total time, count, ref, est)
      est_keys = ("FLOPS", "B/s mem", "B/s lds")
      total = 0
      for dev,e in tagged:
        et = e["dur"] * 1e-3
        t, c, ref, est = agg.get((dev,e["name"]), (0.0, 0, None, {}))
        est.update({k:est.get(k, 0.0)+e["fmt"][k]*e["dur"]*1e-6 for k in est_keys if k in e["fmt"]})
        agg[(dev,e["name"])] = (t+et, c+1, e["ref"], est)
        total += et
      items = sorted(agg.items(), key=lambda kv:kv[1][0], reverse=True)
      num_rows = len(items) if args.t < 0 else args.t
      for (dev,name),(t,c,ref,est) in items[:num_rows]:
        display = f"{dev[:7]:7s} {fmt_colored(name)}" if not args.src else fmt_colored(name)
        yield {"name":display, "dur_ms":t, "count":c, "pct":t/total*100.0, "ref":ref, "fmt":{k:int(est[k]/(t*1e-3)) for k in est_keys if k in est}}
      if num_rows > 0 and items[num_rows:]:
        other_t = sum(t for _,(t,_,_,_) in items[num_rows:])
        other_c = sum(c for _,(_,c,_,_) in items[num_rows:])
        yield {"name":"Other", "dur_ms":other_t, "count":other_c, "pct":other_t/total*100.0, "ref":None, "fmt":None}
    def produce_all_kernels() -> Iterator[dict]:
      event_streams = [[(e["st"], n, e) for e in l["events"]] for n,l in timelines] if not args.src \
                      else [[(e["st"], args.src[0], e) for e in unwrap(data)["events"]]]
      if not args.src:
        for n,l in profile["layout"].items():
          if not isinstance(l, dict) or l.get("event_type") != 0: yield {"device":"SOURCE", "name":n, "st_ms":0, "ref":None, "ext":None}
      marker_stream = sorted([(m["ts"], "MARKER", m) for m in profile.get("markers", [])], key=lambda t:t[0])
      for ts,dev,e in heapq.merge(*event_streams, marker_stream, key=lambda t:t[0]):
        if dev == "MARKER":
          yield {"device":dev, "name":fmt_colored(e["name"]), "st_ms":ts*1e-3, "ref":None, "ext":None}
          continue
        ext, fmt = [], e["fmt"]
        if (tb:=fmt.pop("tb", [])):
          while tb:
            file, lineno, fxn, code = tb.pop()
            line = f"{file.split('/')[-1]}:{lineno} {fxn}"
            if fmt: ext.append(f"{line} {code}")
            elif not file.startswith("<") and not fxn.startswith("<"): fmt["loc"] = line
        yield {"device":dev, "name":fmt_colored(e["name"]), "dur_ms":e["dur"]*1e-3, "st_ms":e["st"]*1e-3, "fmt":fmt, "ref":e["ref"],
               "ext":"\n".join(ext)}
    def fmt_top(k:dict) -> str:
      return f"{fmt_colored(k['name'])}{' ' * max(0, 38-ansilen(k['name']))} {time_to_str(k['dur_ms']*1e-3, w=9)} {k['count']:7d} {k['pct']:6.2f}%"+\
          (" "*4+fmt_data(k['fmt']) if k['fmt'] else "")
    def fmt_all(k:dict) -> str:
      if k["device"] in {"MARKER", "SOURCE"}: return f"--- {k['device']} {k['name']}"+(f"/{k['st_ms']:9.2f}ms" if k['st_ms'] else "")
      ptm = colored(time_to_str(k["dur_ms"]*1e-3, w=9), "yellow" if k["dur_ms"] > 10 else None)
      name = f"*** {k['device'][:7]:7s} "+k["name"]+" "*(46-ansilen(k["name"]))
      return f"{name} tm {ptm}/{k['st_ms']:9.2f}ms"+(f" ({fmt_data(k['fmt'])})" if k["fmt"] else "")
    fmt_row = fmt_top if args.t else fmt_all
    seen_refs:set[int] = set()
    def render_event(k:dict, ls=args.list) -> None:
      if len(args.src) > 1 and ansistrip(k["name"]) not in args.src: return None
      print(emit(k, to_str=fmt_row))
      if k["ref"] is not None and k["ref"] not in seen_refs:
        seen_refs.add(k["ref"])
        for i,s in enumerate(viz_data.ctxs[k["ref"]]["steps"]):
          if DEBUG >= 3 and s["name"] == "View Base AST": print_step(s)
          if DEBUG >= 4 and s["name"] == "View Source": print_step(s)
          if DEBUG >= 5 or ls: print(emit(" "*s["depth"]+s["name"]+(f" - {s['match_count']}" if s.get('match_count', 0) else '')))
          if DEBUG >= 6 or (DEBUG >= 5 and s["name"] == "View Kernel Graph") or (s["name"] in args.src):
            print_step(s, print_graph=True, reconstruct_matches=s["name"] in args.src)
          if DEBUG >= 7: print_step(s, reconstruct_matches=True)
      elif DEBUG >= 3 and k.get("ext"): print(emit(k["ext"]))
    for k in (produce_top_kernels if args.t else produce_all_kernels)(): render_event(k)

def get_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="python -m tinygrad.viz.cli")
  parser.add_argument("-s", "--src", nargs="+", default=[], metavar="NAME", help="Select a data source (default: all)")
  parser.add_argument("--list", "--ls", dest="list", action="store_true", help="List sources")
  parser.add_argument("-t", nargs="?", type=int, const=20, metavar="COUNT", help="Aggregate top kernels (optional count, default 20)")
  parser.add_argument("--profile-path", type=str, metavar="PATH", help="Optional path to profile.pkl (default: latest profile)",
                      default=temp("profile.pkl", append_user=True))
  parser.add_argument("--rewrites-path", type=str, metavar="PATH", help="Optional path to rewrites.pkl (default: latest rewrites)",
                      default=temp("rewrites.pkl", append_user=True))
  parser.add_argument("--json", action="store_true", help="Emit profiler output as JSON")
  return parser

if __name__ == "__main__":
  try: main(get_arg_parser().parse_args())
  except KeyboardInterrupt: pass
