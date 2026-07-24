import unittest
from tinygrad.uop.ops import PatternMatcher, UOp, graph_rewrite, Ops, UPat, BottomUpGate

def assert_not_reached(): assert False, "This function should not be reached"
def gate(): raise BottomUpGate

class TestBottomUpGate(unittest.TestCase):
  def test_basic_bottom_up_gate(self):
    """Test that BottomUpGate stops bottom-up"""
    pm = PatternMatcher([
      (UPat(Ops.ADD), gate),
      (UPat(Ops.MUL), assert_not_reached)
    ])

    a,b,c = UOp.variable("a",0,10), UOp.variable("b",0,10), UOp.variable("c",0,10)
    graph_rewrite((a*a)+(b*c), pm, bottom_up=True)

  def test_bottom_up_gate_with_rewriting(self):
    pm = PatternMatcher([
      (UPat.var("a")+UPat.var("a"), lambda a: 2*a),
      (UPat(Ops.MUL), gate),
      (UPat(Ops.CONST), assert_not_reached)
    ])
    a = UOp.variable("a",0,10)
    graph_rewrite(a+a, pm, bottom_up=True)

if __name__ == "__main__":
  unittest.main()
