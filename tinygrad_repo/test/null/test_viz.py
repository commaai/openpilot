import unittest, decimal, sys, json, contextlib, tempfile, pickle, io
from pathlib import Path
from dataclasses import dataclass
from typing import Generator

from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, TrackedPatternMatcher, graph_rewrite, track_rewrites, profile_matches
from tinygrad.uop.symbolic import sym
from tinygrad.dtype import dtypes
from tinygrad.helpers import colored, ansistrip, flatten, TracingKey, ProfileRangeEvent, ProfileEvent, Context, cpu_events, profile_marker
from tinygrad.helpers import cpu_profile, ProfilePointEvent, unwrap
from tinygrad.device import Buffer

from tinygrad.uop.ops import tracked_keys, tracked_ctxs, uop_fields, active_rewrites, active_group, _name_cnt, RewriteTrace
from tinygrad.viz.serve import load_rewrites, get_full_rewrite, uop_to_json, VizData, get_render
from tinygrad.codegen import to_program_cache
from tinygrad.codegen import to_program

@track_rewrites(name=True)
def exec_rewrite(sink:UOp, pm_lst:list[PatternMatcher], names:None|list[str]=None) -> UOp:
  for i,pm in enumerate(pm_lst):
    sink = graph_rewrite(sink, TrackedPatternMatcher(pm.patterns), name=names[i] if names else None)
  return sink

# small container class for the viz server module
class VizTrace:
  # loader init
  def __init__(self): self._data:VizData|None = None
  @property
  def data(self) -> VizData: return unwrap(self._data)
  def set_data(self) -> None:
    data = VizData(RewriteTrace(tracked_keys.copy(), tracked_ctxs.copy(), uop_fields.copy()))
    load_rewrites(data)
    self._data = data
  # the API
  def list_items(self) -> list[dict]:
    return self.data.ctxs
  def get_details(self, rewrite_idx:int, step:int) -> Generator[dict, None, None]:
    assert len(self.data.trace.rewrites) > rewrite_idx, f"only loaded {len(self.data.trace.rewrites)} traces, expecting at least {rewrite_idx}"
    return get_full_rewrite(self.data, self.data.trace.rewrites[rewrite_idx][step])

@contextlib.contextmanager
def save_viz():
  for lst in [tracked_keys, tracked_ctxs, active_rewrites, active_group, _name_cnt]: lst.clear()
  to_program_cache.clear()
  Buffer.profile_events.clear()
  cpu_events.clear()
  viz = VizTrace()
  with Context(VIZ=-1, TRACK_MATCH_STATS=2, PROFILE=1):
    yield viz
  viz.set_data()

class TestViz(unittest.TestCase):
  def test_simple(self):
    with save_viz() as viz:
      a = UOp.variable("a", 0, 10)
      exec_rewrite((a+0)*1, [sym])
    lst = viz.list_items()
    # VIZ displays rewrites in groups of tracked functions
    self.assertEqual(len(lst), 1)
    # each group has a list of steps
    self.assertEqual(len(lst[0]["steps"]), 1)
    # each step has a list of matches
    self.assertEqual(lst[0]["steps"][0]["match_count"], 2)

  def test_rewrites(self):
    with save_viz() as viz:
      a = UOp.variable("a", 0, 10)
      exec_rewrite(a*1, [sym])
      exec_rewrite(a*2, [sym])
    lst = viz.list_items()
    self.assertEqual(len(lst), 2)
    # names dedup using a counter
    self.assertEqual(lst[0]["name"], "exec_rewrite n1")
    self.assertEqual(lst[1]["name"], "exec_rewrite n2")

  def test_steps(self):
    with save_viz() as viz:
      a = UOp.variable("a", 0, 10)
      exec_rewrite(a+1, [PatternMatcher([]), PatternMatcher([])], ["x", "y"])
    steps = viz.list_items()[0]["steps"]
    # steps can optionally have a name
    self.assertEqual(steps[0]["name"], "x")
    self.assertEqual(steps[1]["name"], "y")

  def test_rewrite_location(self):
    def inner(sink): return graph_rewrite(sink, PatternMatcher([]))
    def outer(sink): return inner(sink)
    with save_viz() as viz:
      outer(UOp.variable("a", 1, 10))
    lst = viz.list_items()
    # step location comes from inner rewrite
    fp, lineno = lst[0]["steps"][0]["loc"]
    self.assertEqual(fp, inner.__code__.co_filename)
    self.assertEqual(lineno, inner.__code__.co_firstlineno)

  def test_exceptions(self):
    # VIZ tracks rewrites up to and including the error
    def count_3(x:UOp):
      assert x.arg <= 3
      return x.replace(arg=x.arg+1)
    err_pm = PatternMatcher([(UPat.cvar("x"), count_3),])
    a = UOp.const(dtypes.int, 1)
    with save_viz() as viz:
      with self.assertRaises(AssertionError): exec_rewrite(a, [err_pm])
    lst = viz.list_items()
    err_step = lst[0]["steps"][0]
    self.assertEqual(err_step["match_count"], 4) # 3 successful rewrites + 1 err

  def test_default_name(self):
    with save_viz() as viz:
      a = UOp.variable("a", 1, 10)
      @track_rewrites()
      def name_default(): return graph_rewrite(a, PatternMatcher([]))
      name_default()
    lst = viz.list_items()
    self.assertEqual(lst[0]["name"], "name_default n1")

  # name can also come from a function that returns a string
  def test_dyn_name_fxn(self):
    with save_viz() as viz:
      @track_rewrites(name=lambda *args,ret,**kwargs: ret.render())
      def name_from_fxn(s:UOp, arg:list|None=None): return graph_rewrite(s, PatternMatcher([]))
      name_from_fxn(UOp.variable("a", 1, 10)+1, arg=["test"])
    lst = viz.list_items()
    # name gets deduped by the function call counter
    self.assertEqual(lst[0]["name"], "(a+1) n1")

  # name can also come from a function that returns a TracingKey
  def test_tracing_key(self):
    with save_viz() as viz:
      @track_rewrites(name=lambda inp,ret: TracingKey("custom_name", (inp,)))
      def test(s:UOp): return graph_rewrite(s, PatternMatcher([]))
      test(UOp.variable("a", 1, 10)+1)
    lst = viz.list_items()
    # NOTE: names from TracingKey do not get deduped
    self.assertEqual(lst[0]["name"], "custom_name")

  def test_nested_track_rewrites(self):
    with save_viz() as viz:
      @track_rewrites(name=lambda x,ret: TracingKey(f"inner fxn for {x.render()}", (ret,)))
      def inner(x:UOp): return graph_rewrite(x, PatternMatcher([]), name="each")
      @track_rewrites(name=lambda *args,ret: f"outer rewrite of {len(args)} inputs")
      def outer(*xs:tuple[UOp, ...]): return graph_rewrite(UOp.sink(*[inner(x) for x in xs]), PatternMatcher([]), name="all")
      items = ["a", "b", "c"]
      outer(*[UOp.variable(x, 1, 10) for x in items])
    lst = viz.list_items()
    # inner calls fall outside the outer call
    self.assertEqual(len(lst), len(items)+1)
    self.assertEqual(lst[0]["name"], f"outer rewrite of {len(items)} inputs n1")
    steps = lst[0]["steps"]
    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0]["name"], "all")
    for i in range(len(items)):
      self.assertEqual(lst[i+1]["name"], f"inner fxn for {items[i]}")
      steps = lst[i+1]["steps"]
      self.assertEqual(len(steps), 1)
      self.assertEqual(steps[0]["name"], "each")

  def test_profile_matches(self):
    with save_viz() as viz:
      @profile_matches
      def nested_function(u:UOp):
        for i in range(2): graph_rewrite(u, PatternMatcher([]), name=f"step {i+1}")

      @track_rewrites()
      def main_rewrite(u:UOp):
        graph_rewrite(u, PatternMatcher([]), name="init")
        nested_function(u)

      main_rewrite(UOp.variable("a", 1, 10)+UOp.variable("b", 1, 10))
    steps = viz.list_items()[0]["steps"]
    self.assertEqual(steps[0]["name"], "init")
    self.assertEqual(steps[1]["name"], "nested_function")
    self.assertEqual(len(steps), 4)

  def test_profile_matches_invalid_arg(self):
    with save_viz():
      @profile_matches
      def invalid_fxn(arg:str): return graph_rewrite(UOp(Ops.SINK), PatternMatcher([]))
      with self.assertRaisesRegex(AssertionError, "invalid match tracing input"):
        invalid_fxn("test")

  def test_colored_label(self):
    # NOTE: dataclass repr prints literal escape codes instead of unicode chars
    @dataclass(frozen=True)
    class TestStruct:
      colored_field: str
    a = UOp(Ops.CUSTOM, arg=TestStruct(colored("xyz", "magenta")+colored("12345", "blue")))
    a2 = uop_to_json(VizData(), a)[id(a)]
    self.assertEqual(ansistrip(a2["label"]), f"CUSTOM\n{TestStruct.__qualname__}(colored_field='xyz12345')")

  def test_colored_label_multiline(self):
    with save_viz() as viz:
      arg = colored("x", "green")+"\n"+colored("y", "red")+colored("z", "yellow")+colored("ww\nw", "magenta")
      src = [Tensor.empty(1).uop for _ in range(10)]
      a = UOp(Ops.CUSTOM, src=tuple(src), arg=arg)
      exec_rewrite(a, [PatternMatcher([])])
    a2 = next(viz.get_details(0, 0))["graph"][id(a)]
    self.assertEqual(ansistrip(a2["label"]), "CUSTOM\nx\nyzww\nw")

  def test_inf_loop(self):
    a = UOp.const(dtypes.int, 3)
    b = UOp.const(dtypes.int, 4)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: x.replace(arg=4)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: x.replace(arg=3)),
    ])
    with save_viz() as viz:
      # use smaller stack limit for faster test (default is 250000)
      with Context(REWRITE_STACK_LIMIT=100): self.assertRaises(RuntimeError, exec_rewrite, a, [pm])
    graphs = flatten(x["graph"].values() for x in viz.get_details(0, 0))
    self.assertEqual(graphs[0], uop_to_json(VizData(), a)[id(a)])
    self.assertEqual(graphs[1], uop_to_json(VizData(), b)[id(b)])
    # fallback to NOOP with the error message
    nop = UOp(Ops.NOOP, arg="infinite loop in fixed_point_rewrite")
    self.assertEqual(graphs[2], uop_to_json(VizData(), nop)[id(nop)])

  def test_const_node_visibility(self):
    with save_viz() as viz:
      a = UOp.variable("a", 0, 10, dtype=dtypes.int)
      z = UOp.const(a.dtype, 0)
      alu = a*z
      exec_rewrite(alu, [sym])
    lst = viz.list_items()
    self.assertEqual(len(lst), 1)
    graphs = [x["graph"] for x in viz.get_details(0, 0)]
    # embed const in the parent node when possible
    self.assertEqual(list(graphs[0]), [id(a), id(alu)])
    self.assertEqual(list(graphs[1]), [id(z)])

  # TODO: DEFINE_VAR (shape ()) now gets wrapped in RESHAPE+EXPAND when broadcast against a shaped operand
  # (due to shared OpMixin._binop using _broadcasted). Either extend viz to fold RESHAPE/EXPAND around
  # DEFINE_VAR/RANGE/SPECIAL the way it does for CONST, or redesign scalar-compiler-op broadcasting.
  @unittest.expectedFailure
  def test_const_reshape_expand_folded(self):
    # CONST->RESHAPE->EXPAND should be folded into the ALU node, not shown as separate RESHAPE/EXPAND nodes
    c = UOp.const(dtypes.float, 1.0, device="CPU", shape=(3,4))  # creates CONST->RESHAPE->EXPAND chain
    a = UOp(Ops.DEFINE_VAR, dtypes.float, arg=("a", 0.0, 10.0))
    alu = a + c
    graph = uop_to_json(VizData(), alu)
    # the RESHAPE and EXPAND nodes from the const should not appear in the graph
    labels = {v["label"].split("\n")[0] for v in graph.values()}
    self.assertNotIn("RESHAPE", labels)
    self.assertNotIn("EXPAND", labels)
    # the CONST should be inlined into the ALU node's label
    alu_label = graph[id(alu)]["label"]
    self.assertIn("CONST", alu_label)

# VIZ displays nested graph_rewrites in a tree view

def leaf_rewrite(x:UOp): return x.rtag(1) if x.tag is None else None
leaf = TrackedPatternMatcher([(UPat(Ops.DEFINE_VAR, name="x"), leaf_rewrite)])

def branch_rewrite(x:UOp, y:UOp):
  if x.tag is not None: return
  x2 = graph_rewrite(x, leaf, name="leaf_left")
  y2 = graph_rewrite(y, leaf, name="leaf_right")
  return x2 * y2
branch = TrackedPatternMatcher([(UPat.var("x")+UPat.var("y"), branch_rewrite)])

def root_rewrite(root:UOp):
  new_src = tuple(graph_rewrite(b, branch, name=f"branch_{i}") for i,b in enumerate(root.src))
  return root.replace(src=new_src)
root = TrackedPatternMatcher([(UPat(Ops.SINK, src=UPat(Ops.ADD), name="root"), root_rewrite),])

class TestVizTree(unittest.TestCase):
  def assertStepEqual(self, step:dict, want:dict):
    for k,v in want.items():
      self.assertEqual(step[k], v, f"failed at '{k}': {v} != {step[k]}\n{step=}")

  def test_tree_view(self):
    with save_viz() as viz:
      a = UOp.variable("a",0,10)
      b = UOp.variable("b",0,10)
      c = UOp.variable("c",0,10)
      d = UOp.variable("d",0,10)
      sink = UOp.sink(a+b, c+d)
      def tree_rewrite(): return graph_rewrite(sink, root, name="root")
      tree_rewrite()
    lst = viz.list_items()
    steps = lst[0]["steps"]
    self.assertEqual(len(steps), 1+2+4)
    self.assertStepEqual(steps[0], {"name":"root", "depth":0, "match_count":1})
    self.assertStepEqual(steps[1], {"name":"branch_0", "depth":1, "match_count":1})
    self.assertStepEqual(steps[2], {"name":"leaf_left", "depth":2, "match_count":1})
    self.assertStepEqual(steps[3], {"name":"leaf_right", "depth":2, "match_count":1})
    self.assertStepEqual(steps[4], {"name":"branch_1", "depth":1, "match_count":1})
    self.assertStepEqual(steps[5], {"name":"leaf_left", "depth":2, "match_count":1})
    self.assertStepEqual(steps[6], {"name":"leaf_right", "depth":2, "match_count":1})

import gc

def bufs_allocated() -> int:
  gc.collect()
  return sum([type(x).__name__ == "Buffer" and type(x).__module__ == "tinygrad.device" for x in gc.get_objects()])

class TestVizGC(unittest.TestCase):
  def test_gc(self):
    with save_viz() as viz:
      init = bufs_allocated()
      a = UOp.new_buffer("NULL", 10, dtypes.char)
      a.buffer.allocate()
      exec_rewrite(a, [PatternMatcher([])])
      del a
      self.assertEqual(bufs_allocated()-init, 0)
    lst = viz.list_items()
    self.assertEqual(len(lst), 1)

  @unittest.skip("it's not generic enough to handle arbitrary UOps in arg")
  def test_gc_uop_in_arg(self):
    with save_viz() as viz:
      init = bufs_allocated()
      a = UOp.new_buffer("NULL", 10, dtypes.char)
      a.buffer.allocate()
      exec_rewrite(UOp(Ops.CUSTOM, src=(a,), arg=a), [PatternMatcher([])])
      del a
      self.assertEqual(bufs_allocated()-init, 0)
    lst = viz.list_items()
    self.assertEqual(len(lst), 1)

# VIZ integrates with other parts of tinygrad

from tinygrad import Tensor, Device, TinyJit, Variable, function

class TestVizIntegration(unittest.TestCase):
  # codegen supports rendering of code blocks
  def test_codegen_tracing(self):
    with save_viz() as viz:
      ast = (Tensor.empty(4)+Tensor.empty(4)).schedule_linear().src[0].src[0]
      prg = to_program(ast, Device[Device.DEFAULT].renderer)
    lst = viz.list_items()
    self.assertEqual(len(lst), 3)
    self.assertEqual(lst[0]["name"], "Callify 1 Buffer n1")
    self.assertEqual(lst[1]["name"], "Schedule 1 Kernel n1")
    self.assertEqual(lst[2]["name"], prg.arg.name)

  # schedule graph CALL nodes have a link to jump to codegen
  def test_link_sched_codegen(self):
    with save_viz() as viz:
      c1 = Tensor.empty(4, device="NULL").add(1)
      c2 = Tensor.empty(8, device="NULL").add(1)
      with Context(SCACHE=0):
        sched = c1.schedule_linear(c2)
      from tinygrad.engine.realize import compile_linear
      sched = compile_linear(sched)
      with Context(NO_COLOR=0):
        prgs = [to_program(si.src[0], Device[c1.device].renderer).arg.name for si in sched.src]
    lst = viz.list_items()
    sched_idx = next(i for i,l in enumerate(lst) if l["name"].startswith("Schedule"))
    viz_kernel = next(i for i,s in enumerate(lst[sched_idx]["steps"]) if s["name"] == "View Kernel Graph")
    with Context(NO_COLOR=1):
      graph = next(viz.get_details(sched_idx, viz_kernel))["graph"]
    call_nodes = [n for n in graph.values() if n["label"].startswith("CALL")]
    for i,n in enumerate(call_nodes):
      assert n["ref"] is not None
      self.assertEqual(lst[n["ref"]]["name"], prgs[i])
      assert ansistrip(prgs[i]) in n["label"], f"CALL must contain kernel name, got {n['label']}"

  def test_link_sched_codegen_beam(self):
    with Context(BEAM=2):
      self.test_link_sched_codegen()

  @Context(TRACEMETA=2)
  def test_metadata_tracing(self):
    with save_viz() as viz:
      a = Tensor.empty(1)
      b = Tensor.empty(1)
      metadata = (alu:=a+b).uop.metadata
      alu.schedule_linear()
    graph = next(viz.get_details(0, 0))["graph"]
    self.assertEqual(len([n for n in graph.values() if repr(metadata) in n["label"]]), 1)

  # tracing also works without a track_rewrites context
  # all graph_rewrites get put into the default group
  def test_default_tracing(self):
    with save_viz() as viz:
      def test(root):
        return graph_rewrite(root, sym)
      test(c:=UOp.const(dtypes.int, 1))
      test(c+1)
    ls = viz.list_items()
    self.assertEqual(len(ls), 1)
    self.assertEqual(ls[0]["name"], "default graph_rewrite")

  # using @track_rewrites organizes function calls into groups
  # and nicely counts function calls.
  def test_group_traces(self):
    with save_viz() as viz:
      @track_rewrites()
      def test(root):
        return graph_rewrite(root, sym)
      test(c:=UOp.const(dtypes.int, 1))
      test(c+1)
    ls = viz.list_items()
    self.assertEqual(len(ls), 2)
    for i in range(2): self.assertEqual(ls[i]["name"], f"test n{i+1}")

  # @track_rewrites always starts a new group.
  def test_group_combined(self):
    with save_viz() as viz:
      def default_test(root): return graph_rewrite(root, sym)
      tracked_test = track_rewrites()(default_test)
      c = UOp.const(dtypes.int, 1)
      default_test(c+1) # goes to the default group
      tracked_test(c)   # all rewrites after this go inside the second group.
      default_test(c+2)
    ls = viz.list_items()
    self.assertEqual(len(ls), 2)
    self.assertEqual(list(next(viz.get_details(0, 0))["graph"]), [id(c+1)])
    self.assertEqual(list(next(viz.get_details(1, 0))["graph"]), [id(c)])
    self.assertEqual(list(next(viz.get_details(1, 1))["graph"]), [id(c+2)])

  def test_recurse(self):
    with save_viz() as viz:
      a = Tensor.empty(10)
      for _ in range(10_000): a += a
      graph_rewrite(a.uop, PatternMatcher([]))
    lst = viz.list_items()
    assert len(lst) == 1

  def test_jit(self):
    with save_viz():
      @TinyJit
      def f(a, b, c): return (a+b).contiguous().mul(3), c.add(1).contiguous().assign(a.to(c.device)), b.assign(c.to(b.device))
      a, b, c = Tensor.empty(16, device="NULL"), Tensor.empty(16, device="NULL"), Tensor.empty(16, device="NULL:1")
      for _ in range(3): Tensor.realize(*f(a, b, c))
    out = load_profile(cpu_events)
    self.assertEqual(["NULL", "NULL Graph", "NULL:SDMA:0", "NULL:1", "NULL:1:SDMA:0"], [k for k in out["layout"] if k.startswith("NULL")])
    self.assertEqual(len(out["layout"]["NULL"]["events"]), 2*3)
    self.assertEqual(len(out["layout"]["NULL:SDMA:0"]["events"]), 3)
    self.assertEqual(len(out["layout"]["NULL Graph"]["events"]), 2)

from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent, ProfileGraphEntry
from tinygrad.viz.serve import get_profile
from tinygrad.viz.cli import decode_profile

def load_profile(lst:list[ProfileEvent]) -> dict: return decode_profile(get_profile(VizData(), lst))

class TestVizProfiler(unittest.TestCase):
  def test_transfer_uses_copy_device(self):
    with save_viz():
      a = Tensor.ones(1, device="NULL").contiguous().realize()
      a.to("NULL:1").realize()
    range_events = [e for e in cpu_events if isinstance(e, ProfileRangeEvent)]
    compute_events = [e for e in range_events if e.device == "NULL"]
    copy_events = [e for e in range_events if e.device.endswith(":SDMA:0")]
    self.assertGreater(len(compute_events), 0, "expected compute events on base device")
    self.assertGreater(len(copy_events), 0, "transfer must produce events with ':SDMA' device suffix")

  def test_node(self):
    prof = [ProfileRangeEvent(device='NV', name='E_2', st=decimal.Decimal(1000), en=decimal.Decimal(1010)),
            ProfileDeviceEvent(device='NV', tdiff=decimal.Decimal(-1000))]

    j = load_profile(prof)

    dev_events = j['layout']['NV']['events']
    self.assertEqual(len(dev_events), 1)
    event = dev_events[0]
    self.assertEqual(event['name'], 'E_2')
    self.assertEqual(event['st'], 0)
    self.assertEqual(event['dur'], 10)
    assert event['ref'] is None

  def test_copy_node(self):
    prof = [ProfileRangeEvent(device='NV:SDMA:0', name='COPYxx', st=decimal.Decimal(1000), en=decimal.Decimal(1010)),
            ProfileRangeEvent(device='NV:2:SDMA:0', name='COPYxx', st=decimal.Decimal(1000), en=decimal.Decimal(1010)),
            ProfileDeviceEvent(device='NV:SDMA:0', tdiff=decimal.Decimal(-100)),
            ProfileDeviceEvent(device='NV:2:SDMA:0', tdiff=decimal.Decimal(-80))]

    j = load_profile(prof)

    event = j['layout']['NV:SDMA:0']['events'][0]
    self.assertEqual(event['name'], 'COPYxx')
    self.assertEqual(event['st'], 0)   # first event
    self.assertEqual(event['dur'], 10)

    event2 = j['layout']['NV:2:SDMA:0']['events'][0]
    self.assertEqual(event2['st'], 20) # second event, diff clock

    self.assertEqual(j["dur"], (event2["st"]+event2["dur"])-event["st"])

  def test_copy_node_bandwidth(self):
    sz = 256*1024*1024
    dur = 10_000
    prof = [ProfileRangeEvent(device='NV:SDMA:0', name=TracingKey("NV -> NV:1", ret=sz), st=decimal.Decimal(1000), en=decimal.Decimal(1000+dur)),
            ProfileDeviceEvent(device='NV:SDMA:0', tdiff=decimal.Decimal(-1000))]
    j = load_profile(prof)
    event = j['layout']['NV:SDMA:0']['events'][0]
    self.assertEqual(event['fmt'], {"B/s": sz/(dur*1e-6), "B": sz})

  def test_graph(self):
    prof = [ProfileDeviceEvent(device='NV', tdiff=decimal.Decimal(-1000)),
            ProfileDeviceEvent(device='NV:1:SDMA:0', tdiff=decimal.Decimal(-50)),
            ProfileGraphEvent(ents=[ProfileGraphEntry(device='NV', name='E_25_4n2', st_id=0, en_id=1),
                                    ProfileGraphEntry(device='NV:1:SDMA:0', name='NV -> NV:1', st_id=2, en_id=3)],
                              deps=[[], [0]],
                              sigs=[decimal.Decimal(1000), decimal.Decimal(1002), decimal.Decimal(1004), decimal.Decimal(1008)])]

    j = load_profile(prof)

    tracks = list(j['layout'])
    self.assertEqual(tracks[0], 'NV')
    self.assertEqual(tracks[1], 'NV Graph')
    self.assertEqual(tracks[2], 'NV:1:SDMA:0')

    nv_events = j['layout']['NV']['events']
    self.assertEqual(nv_events[0]['name'], 'E_25_4n2')
    self.assertEqual(nv_events[0]['st'], 0)
    self.assertEqual(nv_events[0]['dur'], 2)

    sdma_events = j['layout']['NV:1:SDMA:0']['events']
    self.assertEqual(sdma_events[0]['name'], 'NV -> NV:1')
    self.assertEqual(sdma_events[0]['st'], 954)

    graph_events = j['layout']['NV Graph']['events']
    self.assertEqual(graph_events[0]['st'], nv_events[0]['st'])
    self.assertEqual(graph_events[0]['st']+graph_events[0]['dur'], sdma_events[0]['st']+sdma_events[0]['dur'])

  def test_graph_copy_bandwidth(self):
    sz = 256*1024*1024
    dur = 10_000
    prof = [ProfileDeviceEvent(device='NV', tdiff=decimal.Decimal(-1000)),
            ProfileDeviceEvent(device='NV:1:SDMA:0', tdiff=decimal.Decimal(-50)),
            ProfileGraphEvent(ents=[ProfileGraphEntry(device='NV:1:SDMA:0', name=TracingKey("NV -> NV:1", ret=sz), st_id=0, en_id=1)],
                              deps=[[]],
                              sigs=[decimal.Decimal(1004), decimal.Decimal(1004+dur)])]

    j = load_profile(prof)
    sdma_events = j['layout']['NV:1:SDMA:0']['events']
    self.assertEqual(sdma_events[0]["fmt"], {"B/s": sz/(dur*1e-6), "B": sz})

  def test_block_ordering(self):
    prof = [ProfileDeviceEvent(device='NV', tdiff=decimal.Decimal(-1000)),
            ProfileDeviceEvent(device='NV:1', tdiff=decimal.Decimal(-500)),
            ProfileDeviceEvent(device='NV:SDMA:0', tdiff=decimal.Decimal(-100)),
            ProfileRangeEvent(device='NV', name='E_2', st=decimal.Decimal(1000), en=decimal.Decimal(1010)),
            ProfileRangeEvent(device='NV:1', name='E_3', st=decimal.Decimal(1000), en=decimal.Decimal(1010)),
            ProfileRangeEvent(device='NV:SDMA:0', name='COPY', st=decimal.Decimal(1000), en=decimal.Decimal(1010)),
            ProfileGraphEvent(ents=[ProfileGraphEntry(device='NV', name='E_2', st_id=0, en_id=1)],
                              deps=[[]], sigs=[decimal.Decimal(1000), decimal.Decimal(1010)])]
    j = load_profile(prof)
    # graph grouped with its device, memory at the end
    self.assertListEqual(list(j['layout']), ['NV', 'NV Graph', 'NV:SDMA:0', 'NV:1'])

  @unittest.skipIf(sys.platform == 'win32', "TODO: ops_amd import fails on windows")
  def test_multi_sdma_ordering(self):
    props = {"gfx_target_version": 0}
    D, St, En = decimal.Decimal, decimal.Decimal(1000), decimal.Decimal(1010)
    prof = [# 2 AMD GPUs, 2 SDMA engines each
            ProfileDeviceEvent(device='AMD', tdiff=D(-1000), props=props),
            ProfileDeviceEvent(device='AMD:1', tdiff=D(-900), props=props),
            ProfileDeviceEvent(device='AMD:SDMA:0', tdiff=D(-100), props=props),
            ProfileDeviceEvent(device='AMD:SDMA:1', tdiff=D(-80), props=props),
            ProfileDeviceEvent(device='AMD:1:SDMA:0', tdiff=D(-60), props=props),
            ProfileDeviceEvent(device='AMD:1:SDMA:1', tdiff=D(-40), props=props),
            # compute + copy events
            ProfileRangeEvent(device='AMD', name='E_1', st=St, en=En),
            ProfileRangeEvent(device='AMD:1', name='E_2', st=St, en=En),
            ProfileRangeEvent(device='AMD:SDMA:0', name='COPY0', st=St, en=En),
            ProfileRangeEvent(device='AMD:SDMA:1', name='COPY1', st=St, en=En),
            ProfileRangeEvent(device='AMD:1:SDMA:0', name='COPY2', st=St, en=En),
            ProfileRangeEvent(device='AMD:1:SDMA:1', name='COPY3', st=St, en=En),
            # graph spanning compute + copy on GPU 0
            ProfileGraphEvent(ents=[ProfileGraphEntry(device='AMD', name='E_1', st_id=0, en_id=1),
                                    ProfileGraphEntry(device='AMD:SDMA:0', name='COPY0', st_id=2, en_id=3)],
                              deps=[[], [0]], sigs=[St, En, St, En]),
            # memory alloc on both GPUs
            ProfilePointEvent(device='AMD', name='alloc', key=0, arg={"sz":1024, "dtype":dtypes.float}, ts=St),
            ProfilePointEvent(device='AMD:1', name='alloc', key=1, arg={"sz":512, "dtype":dtypes.float}, ts=St)]
    j = load_profile(prof)
    # graph grouped with its device, memory at the end
    self.assertListEqual(list(j['layout']),
      ['AMD', 'AMD Graph', 'AMD:SDMA:0', 'AMD:SDMA:1',
       'AMD:1', 'AMD:1:SDMA:0', 'AMD:1:SDMA:1',
       'AMD Memory', 'AMD:1 Memory'])

  def test_bytes_per_kernel(self):
    step = 10
    n_events = 1_000
    prof = [ProfileRangeEvent("CPU", name="k_test", st=decimal.Decimal(ts:=i*step), en=decimal.Decimal(ts)+step) for i in range(n_events)]
    sz = len(get_profile(VizData(), prof))
    self.assertLessEqual(sz/n_events, 26)

  def test_calltrace(self):
    with save_viz() as viz:
      def fxn(): return Tensor.empty(10).mul(2).realize()
      with cpu_profile(TracingKey("test_fxn"), "CUSTOM"):
        fxn()
    codegen_trace = viz.list_items()[0]["steps"][0]["trace"]
    assert any(fxn.__code__.co_filename == f and fxn.__code__.co_firstlineno == l for f,l,*_ in codegen_trace), str(codegen_trace)
    profile_ret = load_profile(cpu_events)
    e = profile_ret["layout"]["CUSTOM"]["events"][0]
    self.assertEqual(e["name"], "test_fxn")
    runtime_trace = e["fmt"]["tb"]
    assert any(fxn.__code__.co_filename == f and fxn.__code__.co_firstlineno+1 == l for f,l,*_ in runtime_trace), str(runtime_trace)

  # can pack up to 1hr 11 min of trace events
  def test_trace_duration(self):
    dur_mins = 72
    n_events = 1_000
    step = decimal.Decimal(dur_mins*60*1e6//n_events)
    prof = [ProfileRangeEvent("CPU", name="k_test", st=decimal.Decimal(ts:=i*step), en=decimal.Decimal(ts)+step) for i in range(n_events)]
    with self.assertRaisesRegex(ValueError, "timestamp out of range"):
      get_profile(VizData(), prof)

  def test_python_marker(self):
    with save_viz():
      a = Tensor.empty(1, device="NULL")
      b = Tensor.empty(1, device="NULL")
      (a+b).realize()
      profile_marker("test 1")
      (a*b).realize()
      profile_marker("test 2")
    profile_ret = load_profile(cpu_events)
    markers = profile_ret["markers"]
    kernels = profile_ret["layout"]["NULL"]["events"]
    self.assertEqual(len(markers), 2)
    assert kernels[0]["st"] <= markers[0]["ts"] <= kernels[1]["st"]
    assert markers[1]["ts"] >= kernels[1]["st"]+kernels[1]["dur"]

  def test_layout_order(self):
    with save_viz():
      def fn(): return
      for dname in ["TINY", "USER", "TEST:1 N1", "TEST:2 N1", "TEST:1 N2", "TEST:1:ENGINE:0", "TEST:1:ENGINE:0 N1", "TEST:1"]:
        with cpu_profile("fn", dname): fn()
    layout = list(load_profile(cpu_events)["layout"])
    self.assertListEqual(layout[:2], ["USER","TINY"])
    self.assertListEqual(layout[2:], ["TEST:1", "TEST:1 N1", "TEST:1 N2", "TEST:1:ENGINE:0", "TEST:1:ENGINE:0 N1", "TEST:2 N1"])

def _alloc(b:int):
  a = Tensor.empty(b, device="NULL", dtype=dtypes.char)
  a.uop.buffer.allocate()
  return a

class TestVizMemoryLayout(unittest.TestCase):
  def test_double_alloc(self):
    with save_viz():
      a = _alloc(1)
      _b = _alloc(1)
    profile_ret = load_profile(Buffer.profile_events)
    ret = profile_ret["layout"][f"{a.device} Memory"]
    self.assertEqual(ret["peak"], 2)
    self.assertEqual(len(ret["events"]), 4)

  def test_del_once(self):
    with save_viz():
      a = _alloc(1)
      del a
      b = _alloc(1)
    profile_ret = load_profile(Buffer.profile_events)
    ret = profile_ret["layout"][f"{b.device} Memory"]
    self.assertEqual(ret["peak"], 1)
    self.assertEqual(len(ret["events"]), 4)

  def test_alloc_free(self):
    with save_viz():
      a = _alloc(1)
      _b = _alloc(1)
      del a
      c = _alloc(1)
    profile_ret = load_profile(Buffer.profile_events)
    ret = profile_ret["layout"][f"{c.device} Memory"]
    self.assertEqual(ret["peak"], 2)
    self.assertEqual(len(ret["events"]), 6)

  def test_free_last(self):
    with save_viz():
      bufs = []
      for _ in range(3):
        bufs.append(_alloc(1))
        profile_marker("alloc")
      device = bufs[0].device
      while bufs:
        b = bufs.pop()
        del b
        profile_marker("free")
    profile = load_profile(cpu_events+Buffer.profile_events)
    ret = profile["layout"][f"{device} Memory"]
    self.assertEqual(ret["peak"], 3)
    self.assertEqual(len(ret["events"]), 6)
    self.assertEqual(len(profile["markers"]), 6)

  def test_producer_simple(self):
    with save_viz():
      a = Tensor.ones(10, device="NULL")
      Tensor.realize(a.add(1).contiguous())
      b = Tensor.ones(10, device="NULL")
      Tensor.realize(b.add(1).contiguous())
    profile = load_profile(cpu_events+Buffer.profile_events)
    buffers = profile["layout"]["NULL Memory"]["events"]
    programs = profile["layout"]["NULL"]["events"]
    user_cnt = [len(b["arg"]["users"]) for b in buffers if b["arg"].get("users")]
    self.assertEqual(len(user_cnt), len(programs))

  @unittest.skip("flaky")
  def test_inflight_buf(self):
    a = Tensor.empty(1, device="NULL")
    n = 4
    for i in range(n): (a+i).realize()
    profile = load_profile(cpu_events+Buffer.profile_events)
    buffers = profile["layout"]["NULL Memory"]["events"]
    user_cnt = [len(b["arg"]["users"]) for b in buffers if b["arg"].get("users")]
    self.assertEqual(max(user_cnt), n)
    input_buf = buffers.pop()
    assert all(u[3] == 0 for u in input_buf["arg"]["users"])

  def test_annotate_read_write(self):
    with save_viz():
      a = Tensor.ones(4, device="NULL").contiguous().realize()
      b = a.assign(a+2)
      c = a+1
      Tensor.realize(b, c)
    buf_events = load_profile(cpu_events+Buffer.profile_events)["layout"]["NULL Memory"]["events"]
    users = next((b["arg"]["users"] for b in buf_events if len(b["arg"].get("users",[])) == 3))
    self.assertEqual(users[0][3], 1) # write Tensor.ones
    self.assertEqual(users[1][3], 2) # read+write Tensor.assign
    self.assertEqual(users[2][3], 0) # readonly

  def test_dedup_users(self):
    with save_viz():
      a = Tensor.empty(1, device="NULL")
      for _ in range(n:=4): a.add(1).realize()
    profile = load_profile(cpu_events+Buffer.profile_events)
    programs = profile["layout"][a.device]["events"]
    users = profile["layout"][f"{a.device} Memory"]["events"].pop()["arg"]["users"]
    self.assertEqual(len(programs), len(set(users)), n)

from tinygrad.uop.ops import KernelInfo
from tinygrad.renderer.amd.dsl import s
from tinygrad.runtime.autogen.amd.rdna3.ins import (s_add_u32, s_branch, s_cbranch_execz, s_cbranch_scc0, s_cbranch_scc1, s_cmp_eq_i32,
                                                    s_cmp_eq_u64, s_code_end, s_endpgm, s_mov_b32, s_nop)
from extra.gemm.amd_asm_matmul import Kernel

class TestCfg(unittest.TestCase):
  def setUp(self): self.arch = "gfx1100"

  def get_cfg(self, name:str, k:Kernel):
    insts = k.finalize()
    def fxn(out:UOp) -> UOp:
      lidx = UOp.special(1, "lidx0")
      gidx = UOp.special(1, "gidx0")
      sink = UOp.sink(out.base, lidx, gidx, arg=KernelInfo(name=name))
      return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="NULL"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))
    with save_viz() as viz:
      with Context(DEV=f"NULL::{self.arch}"):
        out = Tensor.custom_kernel(Tensor.empty(1), fxn=fxn)[0]
        _ = to_program(out.schedule_linear().src[-1].src[0], Device[out.device].renderer)
    codegen_rewrites = next(s for s in viz.list_items() if s["name"] == name)
    disasm = next(s for s in codegen_rewrites["steps"] if s["name"] == "View Disassembly")
    return get_render(viz.data, disasm["query"])

  def test_simple(self):
    k = Kernel()
    k.label("entry")
    k.emit(s_branch(), target="bb1")
    k.label("bb1")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    cfg = self.get_cfg("simple", k)["data"]
    self.assertEqual(len(cfg["blocks"]), 2)

  def test_diamond(self):
    k = Kernel()
    k.label("entry")
    k.emit(s_mov_b32(s[0], 0))
    k.emit(s_mov_b32(s[1], 0))
    k.emit(s_cmp_eq_u64(s[0:1], 0))
    k.emit(s_cbranch_scc1(), target="if")
    k.emit(s_branch(), target="else")
    k.label("if")
    k.emit(s_nop(1))
    k.emit(s_branch(), target="end")
    k.label("else")
    k.emit(s_nop(0))
    k.label("end")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    ret = self.get_cfg("diamond", k)
    cfg = ret["data"]
    self.assertEqual(len(cfg["blocks"]), 5)
    edge_count = sum(len(v) for v in cfg["paths"].values())
    self.assertEqual(edge_count, 5)
    references:dict[str, list[str]] = {}
    for pc, tokens in cfg["pc_tokens"].items():
      for t in tokens:
        for key in t["keys"]: references.setdefault(key, []).append(pc)
    self.assertEqual(len(references["r0"]), 2)
    insts = [cfg["pc_tokens"][pc][0]["st"] for pc in references["r0"]]
    self.assertEqual(insts, ['s_mov_b32', 's_cmp_eq_u64'])
    end_block = [" ".join(t["st"] for t in cfg["pc_tokens"][pc]) for pc in list(cfg["blocks"].values())[-1]]
    code_line = ret["src"].splitlines()[-1]
    self.assertEqual(len(end_block), 2)
    for st in [end_block[-1], code_line]:
      assert st.startswith("s_code_end") and st.endswith("x)"), st

  def test_loop(self):
    k = Kernel()
    k.label("entry")
    k.emit(s_mov_b32(s[1], 4))
    k.label("loop")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_cbranch_scc0(), target="loop")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    self.get_cfg("simple_loop", k)

  def test_loop_branch(self):
    k = Kernel()
    k.label("entry")
    k.emit(s_mov_b32(s[1], 4))
    k.label("loop")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_cmp_eq_i32(s[1], 2))
    k.emit(s_cbranch_scc1(), target="cond")
    k.emit(s_branch(), target="cont")
    k.label("cond")
    k.emit(s_add_u32(s[1], s[1], -2))
    k.label("cont")
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_cbranch_scc0(), target="loop")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    self.get_cfg("loop_if", k)

  def test_loop_break(self):
    k = Kernel()
    k.label("entry")
    k.emit(s_mov_b32(s[1], 8))
    k.label("loop")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_cmp_eq_i32(s[1], 5))
    k.emit(s_cbranch_scc1(), target="break")
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_cbranch_scc0(), target="loop")
    k.label("break")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    self.get_cfg("loop_break", k)

  def test_switch(self):
    k = Kernel()
    k.label("entry")
    k.emit(s_cmp_eq_i32(s[0], 0))
    k.emit(s_cbranch_scc1(), target="case0")
    k.emit(s_cmp_eq_i32(s[0], 1))
    k.emit(s_cbranch_scc1(), target="case1")
    k.emit(s_branch(), target="case2")
    k.label("case0")
    k.emit(s_nop(0))
    k.emit(s_branch(), target="join")
    k.label("case1")
    k.emit(s_nop(1))
    k.emit(s_branch(), target="join")
    k.label("case2")
    k.emit(s_nop(2))
    k.emit(s_branch(), target="join")
    k.label("join")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    self.get_cfg("switch_case", k)

  def test_ping_pong(self):
    k = Kernel()
    k.label("entry")
    k.emit(s_cmp_eq_i32(s[0], 0))
    k.emit(s_cbranch_scc1(), target="ping")
    k.emit(s_branch(), target="pong")
    k.label("ping")
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_cbranch_scc1(), target="pong")
    k.emit(s_branch(), target="end")
    k.label("pong")
    k.emit(s_cmp_eq_i32(s[2], 0))
    k.emit(s_cbranch_scc1(), target="ping")
    k.label("end")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    self.get_cfg("ping_pong", k)

  def test_colored_blocks(self):
    N = 10
    k = Kernel()
    k.label("entry")
    k.emit(s_branch(), target="init0")
    for i in range(N):
      loop = f"loop{i}"
      k.label(f"init{i}")
      k.emit(s_mov_b32(s[1], i + 1))
      k.emit(s_branch(), target=loop)
      k.label(loop)
      k.emit(s_nop(i & 7))
      k.emit(s_add_u32(s[1], s[1], -1))
      k.emit(s_cmp_eq_i32(s[1], 0))
      k.emit(s_cbranch_scc0(), target=loop)
      k.emit(s_branch(), target=f"init{i+1}" if i + 1 < N else "end")
    k.label("end")
    k.emit(s_endpgm())
    k.emit(s_code_end())
    self.get_cfg("test_colored_blocks", k)

  def test_jump_back_to_end(self):
    k = Kernel()
    k.label("entry")
    k.emit(s_mov_b32(s[1], 2))
    k.emit(s_cbranch_execz(), target="loop")
    k.label("end")
    k.emit(s_endpgm())
    k.label("loop")
    k.emit(s_add_u32(s[1], s[1], -1))
    k.emit(s_cmp_eq_i32(s[1], 0))
    k.emit(s_branch(), target="end")
    k.emit(s_code_end())
    self.get_cfg("jump_back_to_end", k)

# launch viz cli without subprocess
def run_cli(*cli_args) -> list[dict]:
  from tinygrad.viz.cli import main, get_arg_parser
  args = get_arg_parser().parse_args(cli_args+("--json",))
  with contextlib.redirect_stdout(buf:=io.StringIO()):
    main(args)
  return [json.loads(line) for line in buf.getvalue().strip().splitlines()]

@contextlib.contextmanager
def write_files(viz) -> list[str]:
  with tempfile.TemporaryDirectory() as tmpdir:
    (r:=Path(tmpdir)/"rewrites.pkl").write_bytes(pickle.dumps(viz.data.trace))
    (p:=Path(tmpdir)/"profile.pkl").write_bytes(pickle.dumps(cpu_events))
    yield ["--rewrites-path", str(r), "--profile-path", str(p)]

class TestCLI(unittest.TestCase):
  def test_reconstruct_debug(self):
    with save_viz() as viz:
      Tensor.empty(1, device="NULL").add(2.0).realize()
      profile_marker("marker @ 1")
      Tensor.empty(1, device="NULL").add(3.0).realize()
    with write_files(viz) as files, Context(DEBUG=4):
      out = run_cli(*files, "-s", "NULL")
    assert any(s.get("value", "").startswith("void E") for s in out)
    assert any(s.get("name", "") == "marker @ 1" for s in out)

  def test_aggregate(self):
    N, CNT = 1024, 5
    with save_viz() as viz:
      for _ in range(CNT):
        (Tensor.empty(N, N, device="NULL")@Tensor.empty(N, N, device="NULL")).realize()
      for _ in range(CNT):
        (Tensor.empty(N, N, device="NULL").assign(Tensor.empty(N, N, device="NULL"))).realize()
    with write_files(viz) as files, Context(NO_COLOR=1):
      kernels = run_cli(*files, "-s", "NULL", "-t")
    self.assertEqual(len(kernels), 2)
    gemm_summary = [s for s in kernels if s["name"].startswith("r_")][0]
    copy_summary = [s for s in kernels if s["name"].startswith("E_")][0]
    self.assertEqual(gemm_summary["count"], CNT)
    self.assertEqual(copy_summary["count"], CNT)

  def test_flops(self):
    test_n = [(8, 16), (16, 32), (32, 64)]
    with save_viz() as viz:
      @TinyJit
      def f(a, b): return (a@a.T), (b@b.T)
      a = Tensor.empty(64, 64, device="NULL")
      b = Tensor.empty(64, 64, device="NULL")
      for i_val, j_val in test_n:
        i = Variable("i", 1, 64).bind(i_val)
        j = Variable("j", 1, 64).bind(j_val)
        Tensor.realize(*f(a[:i], b[:j]))
    with write_files(viz) as files:
      out = run_cli(*files, "-s", "NULL")
      aggregate = run_cli(*files, "-s", "NULL", "-t")
    self.assertEqual(len(out), 3*2)
    # flops increases as N gets larger
    gflops = [row["fmt"]["FLOPS"] for row in out]
    self.assertGreater(gflops[4], gflops[2])
    self.assertGreater(gflops[5], gflops[3])
    # aggregate flops
    self.assertEqual(len(aggregate), 2)
    agg_gflops = [row["fmt"]["FLOPS"] for row in aggregate]
    assert all(min(gflops) < v < max(gflops) for v in agg_gflops), f"{agg_gflops}"

  def test_dedup(self):
    with save_viz() as viz:
      for _ in range(CNT:=4):
        Tensor.empty(4, device="NULL").add(1).realize()
        Tensor.empty(8, device="NULL").add(1).realize()
    with write_files(viz) as files, Context(NO_COLOR=1):
      name = run_cli(*files, "-s", "NULL")[0]["name"]
      with Context(DEBUG=3):
        select = run_cli(*files, "-s", "NULL", name)
    self.assertEqual(len([s for s in select if s.get("value")]), 1, "debug output was not deduped")
    self.assertEqual(len([s for s in select if s.get("device") == "NULL"]), CNT, f"expected 4 runs for {name}")

  def test_call_graph(self):
    @function(precompile=True)
    def f(x):
      r = x.sum(axis=1).reshape(32, 1).expand(32, 32).contiguous()
      return x + r
    # turn off scache because this test requires a complete schedule rewrite
    with save_viz() as viz, Context(SCACHE=0):
      f(f(Tensor.empty(32, 32, device="NULL"))).realize()
    with write_files(viz) as files, Context(NO_COLOR=1):
      prgs = [s["name"] for s in run_cli(*files, "-s", "NULL")]
      with Context(DEBUG=5):
        out = run_cli(*files, "-s", "TINY")
    i = next(i for i,s in enumerate(out) if s.get("value", "").lstrip() == "View Kernel Graph")
    # next print is the CALL graph, CLI outputs exactly as web in TestVizIntegration.test_link_sched_codegen
    call_nodes = [n for n in out[i+1].values() if n["label"].startswith("CALL")]
    for i,n in enumerate(call_nodes):
      assert prgs[i] in n["label"], f"CALL must contain kernel name, got {n['label']}"

if __name__ == "__main__":
  unittest.main()
