import unittest, decimal, json

from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, TrackedPatternMatcher
from tinygrad.uop.ops import graph_rewrite, track_rewrites, TRACK_MATCH_STATS
from tinygrad.uop.symbolic import sym
from tinygrad.dtype import dtypes

@track_rewrites(name=True)
def exec_rewrite(sink:UOp, pm_lst:list[PatternMatcher], names:None|list[str]=None) -> UOp:
  for i,pm in enumerate(pm_lst):
    sink = graph_rewrite(sink, TrackedPatternMatcher(pm.patterns), name=names[i] if names else None)
  return sink

# real VIZ=1 pickles these tracked values
from tinygrad.viz.serve import get_metadata
from tinygrad.uop.ops import tracked_keys, tracked_ctxs, active_rewrites, _name_cnt
def get_viz_list(): return get_metadata(tracked_keys, tracked_ctxs)

class TestViz(unittest.TestCase):
  def setUp(self):
    # clear the global context
    for lst in [tracked_keys, tracked_ctxs, active_rewrites, _name_cnt]: lst.clear()
    self.tms = TRACK_MATCH_STATS.value
    TRACK_MATCH_STATS.value = 2
  def tearDown(self): TRACK_MATCH_STATS.value = self.tms

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

  # name can also be the first arg
  def test_self_name(self):
    @track_rewrites()
    def name_is_self(s:UOp): return graph_rewrite(s, PatternMatcher([]))
    name_is_self(arg:=UOp.variable("a", 1, 10))
    lst = get_viz_list()
    self.assertEqual(lst[0]["name"], str(arg))

  # name can also come from a function
  def test_dyn_name_fxn(self):
    @track_rewrites(name=lambda a,ret: a.render())
    def name_from_fxn(s:UOp): return graph_rewrite(s, PatternMatcher([]))
    name_from_fxn(UOp.variable("a", 1, 10)+1)
    lst = get_viz_list()
    self.assertEqual(lst[0]["name"], "(a+1) n1")

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

class TestVizTree(TestViz):
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

# VIZ integrates with other parts of tinygrad

from tinygrad import Tensor, Device
from tinygrad.engine.realize import get_program

class TestVizIntegration(TestViz):
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

from tinygrad.device import ProfileDeviceEvent, ProfileRangeEvent, ProfileGraphEvent, ProfileGraphEntry
from tinygrad.viz.serve import to_perfetto

class TextVizProfiler(unittest.TestCase):
  def test_perfetto_node(self):
    prof = [ProfileRangeEvent(device='NV', name='E_2', st=decimal.Decimal(1000), en=decimal.Decimal(1010), is_copy=False),
            ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100))]

    j = json.loads(to_perfetto(prof))

    # Device regs always first
    self.assertEqual(j['traceEvents'][0]['name'], 'process_name')
    self.assertEqual(j['traceEvents'][0]['ph'], 'M')
    self.assertEqual(j['traceEvents'][0]['args']['name'], 'NV')

    self.assertEqual(j['traceEvents'][1]['name'], 'thread_name')
    self.assertEqual(j['traceEvents'][1]['ph'], 'M')
    self.assertEqual(j['traceEvents'][1]['pid'], j['traceEvents'][0]['pid'])
    self.assertEqual(j['traceEvents'][1]['tid'], 0)
    self.assertEqual(j['traceEvents'][1]['args']['name'], 'COMPUTE')

    self.assertEqual(j['traceEvents'][2]['name'], 'thread_name')
    self.assertEqual(j['traceEvents'][2]['ph'], 'M')
    self.assertEqual(j['traceEvents'][2]['pid'], j['traceEvents'][0]['pid'])
    self.assertEqual(j['traceEvents'][2]['tid'], 1)
    self.assertEqual(j['traceEvents'][2]['args']['name'], 'COPY')

    self.assertEqual(j['traceEvents'][3]['name'], 'E_2')
    self.assertEqual(j['traceEvents'][3]['ts'], 0)
    self.assertEqual(j['traceEvents'][3]['dur'], 10)
    self.assertEqual(j['traceEvents'][3]['ph'], 'X')
    self.assertEqual(j['traceEvents'][3]['pid'], j['traceEvents'][0]['pid'])
    self.assertEqual(j['traceEvents'][3]['tid'], 0)

  def test_perfetto_copy_node(self):
    prof = [ProfileRangeEvent(device='NV', name='COPYxx', st=decimal.Decimal(1000), en=decimal.Decimal(1010), is_copy=True),
            ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100))]

    j = json.loads(to_perfetto(prof))

    self.assertEqual(j['traceEvents'][3]['name'], 'COPYxx')
    self.assertEqual(j['traceEvents'][3]['ts'], 900) # diff clock
    self.assertEqual(j['traceEvents'][3]['dur'], 10)
    self.assertEqual(j['traceEvents'][3]['ph'], 'X')
    self.assertEqual(j['traceEvents'][3]['tid'], 1)

  def test_perfetto_graph(self):
    prof = [ProfileDeviceEvent(device='NV', comp_tdiff=decimal.Decimal(-1000), copy_tdiff=decimal.Decimal(-100)),
            ProfileDeviceEvent(device='NV:1', comp_tdiff=decimal.Decimal(-500), copy_tdiff=decimal.Decimal(-50)),
            ProfileGraphEvent(ents=[ProfileGraphEntry(device='NV', name='E_25_4n2', st_id=0, en_id=1, is_copy=False),
                                    ProfileGraphEntry(device='NV:1', name='NV -> NV:1', st_id=2, en_id=3, is_copy=True)],
                              deps=[[], [0]],
                              sigs=[decimal.Decimal(1000), decimal.Decimal(1002), decimal.Decimal(1004), decimal.Decimal(1008)])]

    j = json.loads(to_perfetto(prof))

    # Device regs always first
    self.assertEqual(j['traceEvents'][0]['args']['name'], 'NV')
    self.assertEqual(j['traceEvents'][1]['args']['name'], 'COMPUTE')
    self.assertEqual(j['traceEvents'][2]['args']['name'], 'COPY')
    self.assertEqual(j['traceEvents'][3]['args']['name'], 'NV:1')
    self.assertEqual(j['traceEvents'][4]['args']['name'], 'COMPUTE')
    self.assertEqual(j['traceEvents'][5]['args']['name'], 'COPY')

    self.assertEqual(j['traceEvents'][6]['name'], 'E_25_4n2')
    self.assertEqual(j['traceEvents'][6]['ts'], 0)
    self.assertEqual(j['traceEvents'][6]['dur'], 2)
    self.assertEqual(j['traceEvents'][6]['pid'], j['traceEvents'][0]['pid'])

    self.assertEqual(j['traceEvents'][7]['name'], 'NV -> NV:1')
    self.assertEqual(j['traceEvents'][7]['ts'], 954)
    self.assertEqual(j['traceEvents'][7]['dur'], 4)
    self.assertEqual(j['traceEvents'][7]['pid'], j['traceEvents'][3]['pid'])

if __name__ == "__main__":
  unittest.main()
