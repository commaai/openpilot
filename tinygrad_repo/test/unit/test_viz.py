import unittest, decimal, json
from dataclasses import dataclass

from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, TrackedPatternMatcher
from tinygrad.uop.ops import graph_rewrite, track_rewrites, TRACK_MATCH_STATS
from tinygrad.uop.symbolic import sym
from tinygrad.dtype import dtypes
from tinygrad.helpers import PROFILE, colored, ansistrip, flatten, TracingKey, ProfileRangeEvent
from tinygrad.device import Buffer

@track_rewrites(name=True)
def exec_rewrite(sink:UOp, pm_lst:list[PatternMatcher], names:None|list[str]=None) -> UOp:
  for i,pm in enumerate(pm_lst):
    sink = graph_rewrite(sink, TrackedPatternMatcher(pm.patterns), name=names[i] if names else None)
  return sink

# real VIZ=1 pickles these tracked values
from tinygrad.uop.ops import tracked_keys, tracked_ctxs, uop_fields, active_rewrites, _name_cnt
from tinygrad.viz import serve
serve.contexts = (tracked_keys, tracked_ctxs, uop_fields)
from tinygrad.viz.serve import get_metadata, uop_to_json, get_details
def get_viz_list(): return get_metadata(tracked_keys, tracked_ctxs)

class BaseTestViz(unittest.TestCase):
  def setUp(self):
    # clear the global context
    for lst in [tracked_keys, tracked_ctxs, active_rewrites, _name_cnt]: lst.clear()
    Buffer.profile_events.clear()
    self.tms = TRACK_MATCH_STATS.value
    self.profile = PROFILE.value
    TRACK_MATCH_STATS.value = 2
    PROFILE.value = 1
  def tearDown(self):
    TRACK_MATCH_STATS.value = self.tms
    PROFILE.value = self.profile

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
    @track_rewrites(name=True)
    def outer(sink): return inner(sink)
    outer(UOp.variable("a", 1, 10))
    lst = get_viz_list()
    # step location comes from inner rewrite
    fp, lineno = lst[0]["steps"][0]["loc"]
    self.assertEqual(fp, inner.__code__.co_filename)
    self.assertEqual(lineno, inner.__code__.co_firstlineno)

  def test_exceptions(self):
    # VIZ tracks rewrites up to the error
    def count_3(x:UOp):
      assert x.arg <= 3
      return x.replace(arg=x.arg+1)
    err_pm = PatternMatcher([(UPat.cvar("x"), count_3),])
    a = UOp.const(dtypes.int, 1)
    with self.assertRaises(AssertionError): exec_rewrite(a, [err_pm])
    lst = get_viz_list()
    err_step = lst[0]["steps"][0]
    self.assertEqual(err_step["match_count"], 3)

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

  def test_colored_label(self):
    # NOTE: dataclass repr prints literal escape codes instead of unicode chars
    @dataclass(frozen=True)
    class TestStruct:
      colored_field: str
    a = UOp(Ops.CUSTOM, arg=TestStruct(colored("xyz", "magenta")+colored("12345", "blue")))
    a2 = uop_to_json(a)[id(a)]
    self.assertEqual(ansistrip(a2["label"]), f"CUSTOM\n{TestStruct.__qualname__}(colored_field='xyz12345')")

  def test_inf_loop(self):
    a = UOp.variable('a', 0, 10)
    b = a.replace(op=Ops.DEFINE_REG)
    pm = PatternMatcher([
      (UPat(Ops.DEFINE_VAR, name="x"), lambda x: x.replace(op=Ops.DEFINE_REG)),
      (UPat(Ops.DEFINE_REG, name="x"), lambda x: x.replace(op=Ops.DEFINE_VAR)),
    ])
    with self.assertRaises(RuntimeError): exec_rewrite(a, [pm])
    graphs = flatten(x["graph"].values() for x in get_details(tracked_ctxs[0][0]))
    self.assertEqual(graphs[0], uop_to_json(a)[id(a)])
    self.assertEqual(graphs[1], uop_to_json(b)[id(b)])
    # fallback to NOOP with the error message
    nop = UOp(Ops.NOOP, arg="infinite loop in fixed_point_rewrite")
    self.assertEqual(graphs[2], uop_to_json(nop)[id(nop)])

  def test_const_node_visibility(self):
    a = UOp.variable("a", 0, 10)
    z = UOp.const(dtypes.int, 0)
    alu = a*z
    exec_rewrite(alu, [sym])
    graphs = [x["graph"] for x in get_details(tracked_ctxs[0][0])]
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
    @track_rewrites()
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
  return sum([isinstance(x, Buffer) for x in gc.get_objects()])

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
  # kernelize has a custom name function in VIZ
  def test_kernelize_tracing(self):
    a = Tensor.empty(4, 4)
    Tensor.kernelize(a+1, a+2)
    lst = get_viz_list()
    self.assertEqual(len(lst), 1)
    self.assertEqual(lst[0]["name"], "Schedule 2 Kernels n1")

  # codegen supports rendering of code blocks
  def test_codegen_tracing(self):
    ast = Tensor.schedule(Tensor.empty(4)+Tensor.empty(4))[0].ast
    prg = get_program(ast, Device[Device.DEFAULT].renderer)
    lst = get_viz_list()
    self.assertEqual(len(lst), 2)
    self.assertEqual(lst[0]["name"], "Schedule 1 Kernel n1")
    self.assertEqual(lst[1]["name"], prg.name)

from tinygrad.device import ProfileDeviceEvent, ProfileGraphEvent, ProfileGraphEntry
from tinygrad.viz.serve import get_profile

class TestVizProfiler(unittest.TestCase):
  def test_perfetto_node(self):
    prof = [ProfileRangeEvent(device='NV', name='E_2', st=decimal.Decimal(1000), en=decimal.Decimal(1010), is_copy=False),
            ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100))]

    j = json.loads(get_profile(prof))

    dev_events = j['layout']['NV']['timeline']['shapes']
    self.assertEqual(len(dev_events), 1)
    event = dev_events[0]
    self.assertEqual(event['name'], 'E_2')
    self.assertEqual(event['st'], 0)
    self.assertEqual(event['dur'], 10)

  def test_perfetto_copy_node(self):
    prof = [ProfileRangeEvent(device='NV', name='COPYxx', st=decimal.Decimal(1000), en=decimal.Decimal(1010), is_copy=True),
            ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100))]

    j = json.loads(get_profile(prof))

    event = j['layout']['NV']['timeline']['shapes'][0]
    self.assertEqual(event['name'], 'COPYxx')
    self.assertEqual(event['st'], 900) # diff clock
    self.assertEqual(event['dur'], 10)

  def test_perfetto_graph(self):
    prof = [ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100)),
            ProfileDeviceEvent(device='NV:1', comp_tdiff=decimal.Decimal(-500), copy_tdiff=decimal.Decimal(-50)),
            ProfileGraphEvent(ents=[ProfileGraphEntry(device='NV', name='E_25_4n2', st_id=0, en_id=1, is_copy=False),
                                    ProfileGraphEntry(device='NV:1', name='NV -> NV:1', st_id=2, en_id=3, is_copy=True)],
                              deps=[[], [0]],
                              sigs=[decimal.Decimal(1000), decimal.Decimal(1002), decimal.Decimal(1004), decimal.Decimal(1008)])]

    j = json.loads(get_profile(prof))

    devices = list(j['layout'])
    self.assertEqual(devices[0], 'NV Graph')
    self.assertEqual(devices[1], 'NV')
    self.assertEqual(devices[2], 'NV:1')

    nv_events = j['layout']['NV']['timeline']['shapes']
    self.assertEqual(nv_events[0]['name'], 'E_25_4n2')
    self.assertEqual(nv_events[0]['st'], 0)
    self.assertEqual(nv_events[0]['dur'], 2)
    #self.assertEqual(j['devEvents'][6]['pid'], j['devEvents'][0]['pid'])

    nv1_events = j['layout']['NV:1']['timeline']['shapes']
    self.assertEqual(nv1_events[0]['name'], 'NV -> NV:1')
    self.assertEqual(nv1_events[0]['st'], 954)
    #self.assertEqual(j['devEvents'][7]['pid'], j['devEvents'][3]['pid'])

    graph_events = j['layout']['NV Graph']['timeline']['shapes']
    self.assertEqual(graph_events[0]['st'], nv_events[0]['st'])
    self.assertEqual(graph_events[0]['st']+graph_events[0]['dur'], nv1_events[0]['st']+nv1_events[0]['dur'])

def _alloc(b:int):
  a = Tensor.empty(b, device="NULL", dtype=dtypes.char)
  a.uop.buffer.allocate()
  return a

class TestVizMemoryLayout(BaseTestViz):
  def test_double_alloc(self):
    a = _alloc(1)
    _b = _alloc(1)
    profile_ret = json.loads(get_profile(Buffer.profile_events))
    ret = profile_ret["layout"][a.device]["mem"]
    self.assertEqual(ret["peak"], 2)
    self.assertEqual(ret["shapes"][0]["x"], [0, 2])
    self.assertEqual(ret["shapes"][1]["x"], [1, 2])

  def test_del_once(self):
    a = _alloc(1)
    del a
    b = _alloc(1)
    profile_ret = json.loads(get_profile(Buffer.profile_events))
    ret = profile_ret["layout"][b.device]["mem"]
    self.assertEqual(ret["peak"], 1)
    self.assertEqual(ret["shapes"][0]["x"], [0, 2])
    self.assertEqual(ret["shapes"][1]["x"], [2, 3])
    self.assertEqual(ret["shapes"][0]["y"], [0, 0])
    self.assertEqual(ret["shapes"][1]["y"], [0, 0])

  def test_alloc_free(self):
    a = _alloc(1)
    _b = _alloc(1)
    del a
    c = _alloc(1)
    profile_ret = json.loads(get_profile(Buffer.profile_events))
    ret = profile_ret["layout"][c.device]["mem"]
    self.assertEqual(ret["peak"], 2)
    self.assertEqual(ret["shapes"][0]["x"], [0, 3])
    self.assertEqual(ret["shapes"][1]["x"], [1, 3, 3, 4])
    self.assertEqual(ret["shapes"][0]["y"], [0, 0])
    self.assertEqual(ret["shapes"][1]["y"], [1, 1, 0, 0])
    self.assertEqual(ret["shapes"][2]["x"], [3, 4])
    self.assertEqual(ret["shapes"][2]["y"], [1, 1])

if __name__ == "__main__":
  unittest.main()
