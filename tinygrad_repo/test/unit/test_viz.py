import unittest, decimal, json
from tinygrad.dtype import dtypes
from tinygrad.ops import TRACK_MATCH_STATS, TrackedPatternMatcher, UOp, graph_rewrite, track_rewrites
from tinygrad.codegen.symbolic import symbolic
from tinygrad.ops import tracked_ctxs as contexts, tracked_keys as keys, _name_cnt, _substitute
from tinygrad.device import ProfileDeviceEvent, ProfileRangeEvent, ProfileGraphEvent, ProfileGraphEntry
from tinygrad.viz.serve import get_metadata, uop_to_json, to_perfetto

# NOTE: VIZ tests always use the tracked PatternMatcher instance
symbolic = TrackedPatternMatcher(symbolic.patterns)
substitute = TrackedPatternMatcher(_substitute.patterns)

class TestViz(unittest.TestCase):
  def setUp(self):
    # clear the global context
    contexts.clear()
    keys.clear()
    _name_cnt.clear()
    self.tms = TRACK_MATCH_STATS.value
    TRACK_MATCH_STATS.value = 2
  def tearDown(self): TRACK_MATCH_STATS.value = self.tms

  def test_viz_simple(self):
    a = UOp.variable("a", 0, 10)
    @track_rewrites(named=True)
    def test(sink): return graph_rewrite(sink, symbolic)
    test(a*1)
    ret = get_metadata(keys, contexts)
    self.assertEqual(len(ret), 1)
    key, val = ret[0]
    self.assertEqual(key, "test_1")
    self.assertEqual(val[0]["match_count"], 1)

  def test_track_two_rewrites(self):
    a = UOp.variable("a", 0, 10)
    @track_rewrites(named=True)
    def test(sink): return graph_rewrite(sink, symbolic)
    test((a+a)*1)
    ret = get_metadata(keys, contexts)
    key, val = ret[0]
    self.assertEqual(len(ret), 1)              # one context
    self.assertEqual(len(val), 1)              # one graph_rewrite call in context
    self.assertEqual(key, "test_1")
    self.assertEqual(val[0]["match_count"], 2) # two upats applied

  def test_track_multiple_calls_one_ctx(self):
    a = UOp.variable("a", 0, 10)
    @track_rewrites(named=True)
    def test(a, b):
      a = graph_rewrite(a, symbolic)
      b = graph_rewrite(b, symbolic)
    test(a*1, a*5)
    ret = get_metadata(keys, contexts)
    key, val = ret[0]
    self.assertEqual(len(ret), 1)              # one context
    self.assertEqual(len(val), 2)              # two graph_rewrite calls in context
    self.assertEqual(key, "test_1")
    self.assertEqual(val[0]["match_count"], 1) # one rewrite for a*0
    self.assertEqual(val[1]["match_count"], 0) # no rewrites for a*5

  def test_track_rewrites(self):
    @track_rewrites(named=True)
    def do_rewrite(x:UOp): return graph_rewrite(x, symbolic)
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 4)
    do_rewrite(a*1)
    do_rewrite(a*b)
    ret = get_metadata(keys, contexts)
    self.assertEqual(len(ret), 2)
    key, m = ret[0]
    self.assertEqual(key, "do_rewrite_1")
    self.assertEqual(m[0]["match_count"], 1)
    key, m = ret[1]
    self.assertEqual(key, "do_rewrite_2")
    self.assertEqual(m[0]["match_count"], 0)

  def test_track_rewrites_with_exception(self):
    @track_rewrites()
    def do_rewrite(x:UOp):
      x = graph_rewrite(x, symbolic) # NOTE: viz tracks this
      raise Exception("test")
    a = UOp.variable("a", 0, 10)
    with self.assertRaises(Exception): do_rewrite(a*1)
    ret = get_metadata(keys, contexts)
    self.assertEqual(len(ret), 1)

  def test_track_rewrites_name_fxn(self):
    @track_rewrites(name_fxn=lambda r: f"output_{r}")
    def do_rewrite(x:UOp):
      x = graph_rewrite(x, symbolic)
      return x.render()
    expr = UOp.variable("a",0,10)*UOp.variable("b",0,10)
    do_rewrite(expr)
    key = get_metadata(keys, contexts)[0][0]
    self.assertEqual(key, "output_(a*b) n1")

    expr2 = UOp.variable("a",0,10)+UOp.variable("b",0,10)
    do_rewrite(expr2)
    key = get_metadata(keys, contexts)[1][0]
    self.assertEqual(key, "output_(a+b) n2")

  # NOTE: CONST UOps do not get nodes in the graph
  def test_dont_create_const_nodes(self):
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 4)
    self.assertEqual(len(uop_to_json(a*1)), 2)
    self.assertEqual(len(uop_to_json(a*b)), 3)

  def test_bottom_up_rewrite(self):
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    c = UOp.variable("c", 0, 10)
    @track_rewrites(named=True)
    def fxn(sink): return graph_rewrite(sink, substitute, ctx={a+b:c}, bottom_up=True)
    fxn(a+b)
    #UOp.substitute(a+b, {a+b:c})
    ret = get_metadata(keys, contexts)
    self.assertEqual(len(ret), 1)
    _, m = ret[0]
    self.assertEqual(m[0]["match_count"], 1)

  # NOTE: calling graph_rewrite when the function isn't decorated with track_rewrites should not VIZ
  def test_rewrite_without_context(self):
    def untracked_graph_rewrite(sink): return graph_rewrite(sink, symbolic)
    @track_rewrites(named=True)
    def tracked_graph_rewrite(sink): return graph_rewrite(sink, symbolic)
    # test
    add = UOp.const(dtypes.int, 2) + UOp.const(dtypes.int, 1)
    untracked_graph_rewrite(add)
    self.assertEqual(len(contexts), 0)
    tracked_graph_rewrite(add)
    self.assertEqual(len(contexts), 1)

  def test_inner_rewrite_location(self):
    # inner rewrite gets tracked in another context
    def inner_rewrite(sink): return graph_rewrite(sink, symbolic)
    @track_rewrites(named=True)
    def tracked_graph_rewrite(sink): return inner_rewrite(sink)
    # test
    add = UOp.const(dtypes.int, 2) + UOp.const(dtypes.int, 1)
    tracked_graph_rewrite(add)
    self.assertEqual(len(contexts), 1)
    # location of context is inner_rewrite
    fp, lineno = contexts[0][0].loc
    self.assertEqual(lineno, inner_rewrite.__code__.co_firstlineno)
    self.assertEqual(fp, inner_rewrite.__code__.co_filename)

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
