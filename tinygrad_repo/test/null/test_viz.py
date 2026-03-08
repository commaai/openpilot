import unittest, decimal, json, struct
from dataclasses import dataclass
from typing import Generator

from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, TrackedPatternMatcher, graph_rewrite, track_rewrites, TRACK_MATCH_STATS, profile_matches
from tinygrad.uop.symbolic import sym
from tinygrad.dtype import dtypes
from tinygrad.helpers import PROFILE, colored, ansistrip, flatten, TracingKey, ProfileRangeEvent, ProfileEvent, Context, cpu_events, profile_marker
from tinygrad.helpers import VIZ, cpu_profile
from tinygrad.device import Buffer

@track_rewrites(name=True)
def exec_rewrite(sink:UOp, pm_lst:list[PatternMatcher], names:None|list[str]=None) -> UOp:
  for i,pm in enumerate(pm_lst):
    sink = graph_rewrite(sink, TrackedPatternMatcher(pm.patterns), name=names[i] if names else None)
  return sink

# real VIZ=1 loads the trace from a file, we just keep it in memory for tests
from tinygrad.uop.ops import tracked_keys, tracked_ctxs, uop_fields, active_rewrites, _name_cnt, RewriteTrace
from tinygrad.viz import serve
serve.trace = RewriteTrace(tracked_keys, tracked_ctxs, uop_fields)
from tinygrad.viz.serve import get_rewrites, get_full_rewrite, uop_to_json
def get_viz_list(): return get_rewrites(serve.trace)
def get_viz_details(rewrite_idx:int, step:int) -> Generator[dict, None, None]:
  lst = get_viz_list()
  assert len(lst) > rewrite_idx, "only loaded {len(lst)} traces, expecting at least {idx}"
  return get_full_rewrite(tracked_ctxs[rewrite_idx][step])

class BaseTestViz(unittest.TestCase):
  def setUp(self):
    # clear the global context
    for lst in [tracked_keys, tracked_ctxs, active_rewrites, _name_cnt]: lst.clear()
    Buffer.profile_events.clear()
    cpu_events.clear()
    self.tms = TRACK_MATCH_STATS.value
    self.profile = PROFILE.value
    self.viz = VIZ.value
    TRACK_MATCH_STATS.value = 2
    PROFILE.value = 1
    VIZ.value = 1
  def tearDown(self):
    TRACK_MATCH_STATS.value = self.tms
    PROFILE.value = self.profile
    VIZ.value = self.viz

class TestViz(BaseTestViz):
  def test_simple(self):
    a = UOp.variable("a", 0, 10)
    exec_rewrite((a+0)*1, [sym])
    lst = get_viz_list()
    # VIZ displays rewrites in groups of tracked functions
    self.assertEqual(len(lst), 1)
    # each group has a list of steps
    self.assertEqual(len(lst[0]["steps"]), 1)
    # each step has a list of matches
    self.assertEqual(lst[0]["steps"][0]["match_count"], 2)

  def test_rewrites(self):
    a = UOp.variable("a", 0, 10)
    exec_rewrite(a*1, [sym])
    exec_rewrite(a*2, [sym])
    lst = get_viz_list()
    self.assertEqual(len(lst), 2)
    # names dedup using a counter
    self.assertEqual(lst[0]["name"], "exec_rewrite n1")
    self.assertEqual(lst[1]["name"], "exec_rewrite n2")

  def test_steps(self):
    a = UOp.variable("a", 0, 10)
    exec_rewrite(a+1, [PatternMatcher([]), PatternMatcher([])], ["x", "y"])
    steps = get_viz_list()[0]["steps"]
    # steps can optionally have a name
    self.assertEqual(steps[0]["name"], "x")
    self.assertEqual(steps[1]["name"], "y")

  def test_rewrite_location(self):
    def inner(sink): return graph_rewrite(sink, PatternMatcher([]))
    def outer(sink): return inner(sink)
    outer(UOp.variable("a", 1, 10))
    lst = get_viz_list()
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
    with self.assertRaises(AssertionError): exec_rewrite(a, [err_pm])
    lst = get_viz_list()
    err_step = lst[0]["steps"][0]
    self.assertEqual(err_step["match_count"], 4) # 3 successful rewrites + 1 err

  def test_default_name(self):
    a = UOp.variable("a", 1, 10)
    @track_rewrites()
    def name_default(): return graph_rewrite(a, PatternMatcher([]))
    name_default()
    lst = get_viz_list()
    self.assertEqual(lst[0]["name"], "name_default n1")

  # name can also come from a function that returns a string
  def test_dyn_name_fxn(self):
    @track_rewrites(name=lambda *args,ret,**kwargs: ret.render())
    def name_from_fxn(s:UOp, arg:list|None=None): return graph_rewrite(s, PatternMatcher([]))
    name_from_fxn(UOp.variable("a", 1, 10)+1, arg=["test"])
    lst = get_viz_list()
    # name gets deduped by the function call counter
    self.assertEqual(lst[0]["name"], "(a+1) n1")

  # name can also come from a function that returns a TracingKey
  def test_tracing_key(self):
    @track_rewrites(name=lambda inp,ret: TracingKey("custom_name", (inp,)))
    def test(s:UOp): return graph_rewrite(s, PatternMatcher([]))
    test(UOp.variable("a", 1, 10)+1)
    lst = get_viz_list()
    # NOTE: names from TracingKey do not get deduped
    self.assertEqual(lst[0]["name"], "custom_name")

  def test_profile_matches(self):
    @profile_matches
    def nested_function(u:UOp):
      for i in range(2): graph_rewrite(u, PatternMatcher([]), name=f"step {i+1}")

    @track_rewrites()
    def main_rewrite(u:UOp):
      graph_rewrite(u, PatternMatcher([]), name="init")
      nested_function(u)

    main_rewrite(UOp.variable("a", 1, 10)+UOp.variable("b", 1, 10))
    steps = get_viz_list()[0]["steps"]
    self.assertEqual(steps[0]["name"], "init")
    self.assertEqual(steps[1]["name"], "nested_function")
    self.assertEqual(len(steps), 4)

  def test_profile_matches_invalid_arg(self):
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
    a2 = uop_to_json(a)[id(a)]
    self.assertEqual(ansistrip(a2["label"]), f"CUSTOM\n{TestStruct.__qualname__}(colored_field='xyz12345')")

  def test_colored_label_multiline(self):
    arg = colored("x", "green")+"\n"+colored("y", "red")+colored("z", "yellow")+colored("ww\nw", "magenta")
    src = [Tensor.empty(1).uop for _ in range(10)]
    a = UOp(Ops.CUSTOM, src=tuple(src), arg=arg)
    exec_rewrite(a, [PatternMatcher([])])
    a2 = next(get_viz_details(0, 0))["graph"][id(a)]
    self.assertEqual(ansistrip(a2["label"]), "CUSTOM\nx\nyzww\nw")

  def test_inf_loop(self):
    a = UOp.const(dtypes.int, 3)
    b = UOp.const(dtypes.int, 4)
    pm = PatternMatcher([
      (UPat(Ops.CONST, arg=3, name="x"), lambda x: x.replace(arg=4)),
      (UPat(Ops.CONST, arg=4, name="x"), lambda x: x.replace(arg=3)),
    ])
    # use smaller stack limit for faster test (default is 250000)
    with Context(REWRITE_STACK_LIMIT=100): self.assertRaises(RuntimeError, exec_rewrite, a, [pm])
    graphs = flatten(x["graph"].values() for x in get_viz_details(0, 0))
    self.assertEqual(graphs[0], uop_to_json(a)[id(a)])
    self.assertEqual(graphs[1], uop_to_json(b)[id(b)])
    # fallback to NOOP with the error message
    nop = UOp(Ops.NOOP, arg="infinite loop in fixed_point_rewrite")
    self.assertEqual(graphs[2], uop_to_json(nop)[id(nop)])

  def test_const_node_visibility(self):
    a = UOp.variable("a", 0, 10, dtype=dtypes.int)
    z = UOp.const(a.dtype, 0)
    alu = a*z
    exec_rewrite(alu, [sym])
    lst = get_viz_list()
    self.assertEqual(len(lst), 1)
    graphs = [x["graph"] for x in get_viz_details(0, 0)]
    # embed const in the parent node when possible
    self.assertEqual(list(graphs[0]), [id(a), id(alu)])
    self.assertEqual(list(graphs[1]), [id(z)])

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

class TestVizTree(BaseTestViz):
  def assertStepEqual(self, step:dict, want:dict):
    for k,v in want.items():
      self.assertEqual(step[k], v, f"failed at '{k}': {v} != {step[k]}\n{step=}")

  def test_tree_view(self):
    a = UOp.variable("a",0,10)
    b = UOp.variable("b",0,10)
    c = UOp.variable("c",0,10)
    d = UOp.variable("d",0,10)
    sink = UOp.sink(a+b, c+d)
    def tree_rewrite(): return graph_rewrite(sink, root, name="root")
    tree_rewrite()
    lst = get_viz_list()
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

class TestVizGC(BaseTestViz):
  def test_gc(self):
    init = bufs_allocated()
    a = UOp.new_buffer("NULL", 10, dtypes.char)
    a.buffer.allocate()
    exec_rewrite(a, [PatternMatcher([])])
    del a
    self.assertEqual(bufs_allocated()-init, 0)
    lst = get_viz_list()
    self.assertEqual(len(lst), 1)

  @unittest.skip("it's not generic enough to handle arbitrary UOps in arg")
  def test_gc_uop_in_arg(self):
    init = bufs_allocated()
    a = UOp.new_buffer("NULL", 10, dtypes.char)
    a.buffer.allocate()
    exec_rewrite(UOp(Ops.CUSTOM, src=(a,), arg=a), [PatternMatcher([])])
    del a
    self.assertEqual(bufs_allocated()-init, 0)
    lst = get_viz_list()
    self.assertEqual(len(lst), 1)

# VIZ integrates with other parts of tinygrad

from tinygrad import Tensor, Device
from tinygrad.engine.realize import get_program

class TestVizIntegration(BaseTestViz):
  # codegen supports rendering of code blocks
  def test_codegen_tracing(self):
    ast = Tensor.schedule(Tensor.empty(4)+Tensor.empty(4))[0].ast
    prg = get_program(ast, Device[Device.DEFAULT].renderer)
    lst = get_viz_list()
    self.assertEqual(len(lst), 2)
    self.assertEqual(lst[0]["name"], "Schedule 1 Kernel n1")
    self.assertEqual(lst[1]["name"], prg.name)

  def test_metadata_tracing(self):
    with Context(TRACEMETA=2):
      a = Tensor.empty(1)
      b = Tensor.empty(1)
      metadata = (alu:=a+b).uop.metadata
      alu.schedule()
      graph = next(get_viz_details(0, 0))["graph"]
    self.assertEqual(len([n for n in graph.values() if repr(metadata) in n["label"]]), 1)

  # tracing also works without a track_rewrites context
  # all graph_rewrites get put into the default group
  def test_default_tracing(self):
    def test(root):
      return graph_rewrite(root, sym)
    test(c:=UOp.const(dtypes.int, 1))
    test(c+1)
    ls = get_viz_list()
    self.assertEqual(len(ls), 1)
    self.assertEqual(ls[0]["name"], "default graph_rewrite")

  # using @track_rewrites organizes function calls into groups
  # and nicely counts function calls.
  def test_group_traces(self):
    @track_rewrites()
    def test(root):
      return graph_rewrite(root, sym)
    test(c:=UOp.const(dtypes.int, 1))
    test(c+1)
    ls = get_viz_list()
    self.assertEqual(len(ls), 2)
    for i in range(2): self.assertEqual(ls[i]["name"], f"test n{i+1}")

  # @track_rewrites always starts a new group.
  def test_group_combined(self):
    def default_test(root): return graph_rewrite(root, sym)
    tracked_test = track_rewrites()(default_test)
    c = UOp.const(dtypes.int, 1)
    default_test(c+1) # goes to the default group
    tracked_test(c)   # all rewrites after this go inside the second group.
    default_test(c+2)
    ls = get_viz_list()
    self.assertEqual(len(ls), 2)
    self.assertEqual(list(next(get_viz_details(0, 0))["graph"]), [id(c+1)])
    self.assertEqual(list(next(get_viz_details(1, 0))["graph"]), [id(c)])
    self.assertEqual(list(next(get_viz_details(1, 1))["graph"]), [id(c+2)])

  def test_recurse(self):
    a = Tensor.empty(10)
    for _ in range(10_000): a += a
    graph_rewrite(a.uop, PatternMatcher([]))
    lst = get_viz_list()
    assert len(lst) == 1

from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent, ProfileGraphEntry
from tinygrad.viz.serve import get_profile

class TinyUnpacker:
  def __init__(self, buf): self.buf, self.offset = buf, 0
  def __call__(self, fmt:str) -> tuple:
    ret = struct.unpack_from(fmt, self.buf, self.offset)
    self.offset += struct.calcsize(fmt)
    return ret

# 0 means None, otherwise it's an enum value
def option(i:int) -> int|None: return None if i == 0 else i-1

def load_profile(lst:list[ProfileEvent]) -> dict:
  ret = get_profile(lst)
  u = TinyUnpacker(ret)
  total_dur, global_peak, index_len, layout_len = u("<IQII")
  strings, dtypes, markers = json.loads(ret[u.offset:u.offset+index_len]).values()
  u.offset += index_len
  layout:dict[str, dict] = {}
  for _ in range(layout_len):
    klen = u("<B")[0]
    k = ret[u.offset:u.offset+klen].decode()
    u.offset += klen
    layout[k] = v = {"events":[]}
    event_type, event_count = u("<BI")
    if event_type == 0:
      for _ in range(event_count):
        name, ref, key, st, dur, fmt = u("<IIIIfI")
        v["events"].append({"name":strings[name], "ref":option(ref), "key":option(key), "st":st, "dur":dur, "fmt":strings[fmt]})
    else:
      v["peak"] = u("<Q")[0]
      for _ in range(event_count):
        alloc, ts, key = u("<BII")
        if alloc: v["events"].append({"event":"alloc", "ts":ts, "key":key, "arg": {"dtype":strings[u("<I")[0]], "sz":u("<Q")[0]}})
        else: v["events"].append({"event":"free", "ts":ts, "key":key, "arg": {"users":[u("<IIIB") for _ in range(u("<I")[0])]}})
  return {"dur":total_dur, "peak":global_peak, "layout":layout, "markers":markers}

class TestVizProfiler(BaseTestViz):
  def test_node(self):
    prof = [ProfileRangeEvent(device='NV', name='E_2', st=decimal.Decimal(1000), en=decimal.Decimal(1010), is_copy=False),
            ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100))]

    j = load_profile(prof)

    dev_events = j['layout']['NV']['events']
    self.assertEqual(len(dev_events), 1)
    event = dev_events[0]
    self.assertEqual(event['name'], 'E_2')
    self.assertEqual(event['st'], 0)
    self.assertEqual(event['dur'], 10)
    assert event['ref'] is None

  def test_copy_node(self):
    prof = [ProfileRangeEvent(device='NV', name='COPYxx', st=decimal.Decimal(1000), en=decimal.Decimal(1010), is_copy=True),
            ProfileRangeEvent(device='NV:2', name='COPYxx', st=decimal.Decimal(1000), en=decimal.Decimal(1010), is_copy=True),
            ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100)),
            ProfileDeviceEvent(device='NV:2', comp_tdiff=decimal.Decimal(-800), copy_tdiff=decimal.Decimal(-80))]

    j = load_profile(prof)

    event = j['layout']['NV']['events'][0]
    self.assertEqual(event['name'], 'COPYxx')
    self.assertEqual(event['st'], 0)   # first event
    self.assertEqual(event['dur'], 10)

    event2 = j['layout']['NV:2']['events'][0]
    self.assertEqual(event2['st'], 20) # second event, diff clock

    self.assertEqual(j["dur"], (event2["st"]+event2["dur"])-event["st"])

  def test_graph(self):
    prof = [ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100)),
            ProfileDeviceEvent(device='NV:1', comp_tdiff=decimal.Decimal(-500), copy_tdiff=decimal.Decimal(-50)),
            ProfileGraphEvent(ents=[ProfileGraphEntry(device='NV', name='E_25_4n2', st_id=0, en_id=1, is_copy=False),
                                    ProfileGraphEntry(device='NV:1', name='NV -> NV:1', st_id=2, en_id=3, is_copy=True)],
                              deps=[[], [0]],
                              sigs=[decimal.Decimal(1000), decimal.Decimal(1002), decimal.Decimal(1004), decimal.Decimal(1008)])]

    j = load_profile(prof)

    tracks = list(j['layout'])
    self.assertEqual(tracks[0], 'NV')
    self.assertEqual(tracks[1], 'NV Graph')
    self.assertEqual(tracks[2], 'NV:1')

    nv_events = j['layout']['NV']['events']
    self.assertEqual(nv_events[0]['name'], 'E_25_4n2')
    self.assertEqual(nv_events[0]['st'], 0)
    self.assertEqual(nv_events[0]['dur'], 2)
    #self.assertEqual(j['devEvents'][6]['pid'], j['devEvents'][0]['pid'])

    nv1_events = j['layout']['NV:1']['events']
    self.assertEqual(nv1_events[0]['name'], 'NV -> NV:1')
    self.assertEqual(nv1_events[0]['st'], 954)
    #self.assertEqual(j['devEvents'][7]['pid'], j['devEvents'][3]['pid'])

    graph_events = j['layout']['NV Graph']['events']
    self.assertEqual(graph_events[0]['st'], nv_events[0]['st'])
    self.assertEqual(graph_events[0]['st']+graph_events[0]['dur'], nv1_events[0]['st']+nv1_events[0]['dur'])

  def test_bytes_per_kernel(self):
    step = 10
    n_events = 1_000
    prof = [ProfileRangeEvent("CPU", name="k_test", st=decimal.Decimal(ts:=i*step), en=decimal.Decimal(ts)+step) for i in range(n_events)]
    sz = len(get_profile(prof))
    self.assertLessEqual(sz/n_events, 26)

  def test_calltrace(self):
    def fxn(): return Tensor.empty(10).mul(2).realize()
    fxn()
    trace = get_viz_list()[0]["steps"][0]["trace"]
    assert any(fxn.__code__.co_filename == f and fxn.__code__.co_firstlineno == l for f,l,*_ in trace), str(trace)

  # can pack up to 1hr 11 min of trace events
  def test_trace_duration(self):
    dur_mins = 72
    n_events = 1_000
    step = decimal.Decimal(dur_mins*60*1e6//n_events)
    prof = [ProfileRangeEvent("CPU", name="k_test", st=decimal.Decimal(ts:=i*step), en=decimal.Decimal(ts)+step) for i in range(n_events)]
    with self.assertRaises(struct.error):
      get_profile(prof)

  def test_python_marker(self):
    with Context(VIZ=1):
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
    def fn(): return
    for dname in ["TINY", "USER", "TEST:1 N1", "TEST:2 N1", "TEST:1 N2"]:
      with cpu_profile("fn", dname): fn()
    layout = list(load_profile(cpu_events)["layout"])
    self.assertListEqual(layout[:2], ["USER","TINY"])
    self.assertListEqual(layout[2:], ["TEST:1 N1","TEST:1 N2", "TEST:2 N1"])

def _alloc(b:int):
  a = Tensor.empty(b, device="NULL", dtype=dtypes.char)
  a.uop.buffer.allocate()
  return a

class TestVizMemoryLayout(BaseTestViz):
  def test_double_alloc(self):
    a = _alloc(1)
    _b = _alloc(1)
    profile_ret = load_profile(Buffer.profile_events)
    ret = profile_ret["layout"][f"{a.device} Memory"]
    self.assertEqual(ret["peak"], 2)
    self.assertEqual(len(ret["events"]), 4)

  def test_del_once(self):
    a = _alloc(1)
    del a
    b = _alloc(1)
    profile_ret = load_profile(Buffer.profile_events)
    ret = profile_ret["layout"][f"{b.device} Memory"]
    self.assertEqual(ret["peak"], 1)
    self.assertEqual(len(ret["events"]), 4)

  def test_alloc_free(self):
    a = _alloc(1)
    _b = _alloc(1)
    del a
    c = _alloc(1)
    profile_ret = load_profile(Buffer.profile_events)
    ret = profile_ret["layout"][f"{c.device} Memory"]
    self.assertEqual(ret["peak"], 2)
    self.assertEqual(len(ret["events"]), 6)

  def test_free_last(self):
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
    a = Tensor.ones(10, device="NULL")
    Tensor.realize(a.add(1).contiguous())
    b = Tensor.ones(10, device="NULL")
    Tensor.realize(b.add(1).contiguous())
    profile = load_profile(cpu_events+Buffer.profile_events)
    buffers = profile["layout"]["NULL Memory"]["events"]
    programs = profile["layout"]["NULL"]["events"]
    user_cnt = [len(b["arg"]["users"]) for b in buffers if b["arg"].get("users")]
    self.assertEqual(len(user_cnt), len(programs))

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
    a = Tensor.empty(1, device="NULL")
    for _ in range(n:=4): a.add(1).realize()
    profile = load_profile(cpu_events+Buffer.profile_events)
    programs = profile["layout"][a.device]["events"]
    users = profile["layout"][f"{a.device} Memory"]["events"].pop()["arg"]["users"]
    self.assertEqual(len(programs), len(set(users)), n)

if __name__ == "__main__":
  unittest.main()
