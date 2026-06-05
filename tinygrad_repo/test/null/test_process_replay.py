import unittest
from tinygrad import Tensor, Device, Context
from tinygrad.codegen import do_to_program
from tinygrad.codegen.opt import Opt, OptOps
from test.external.process_replay.process_replay import replay_to_program
from test.helpers import replace_opts

N = 16
class TestProcessReplay(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ast = (Tensor.empty(N, N) @ Tensor.empty(N, N)).schedule_linear().src[-1].src[0]
    cls.renderer = Device[Device.DEFAULT].renderer

  def test_replay_no_opts(self):
    # opts=None means use default heuristic path
    p = do_to_program(self.ast, self.renderer)
    good, compare, _ = replay_to_program(p, self.ast, self.renderer)
    self.assertEqual(good, compare)

  def test_replay_empty_opts(self):
    # opts=[] means explicitly apply zero opts (unoptimized)
    ast = replace_opts(self.ast, [])
    p = do_to_program(ast, self.renderer)
    good, compare, _ = replay_to_program(p, ast, self.renderer)
    self.assertEqual(good, compare)

  def test_replay_with_opt(self):
    # opts=[Opt(...)] means apply a specific opt
    opts = [Opt(OptOps.UPCAST, 0, 4)]
    ast = replace_opts(self.ast, opts)
    p = do_to_program(ast, self.renderer)
    good, compare, _ = replay_to_program(p, ast, self.renderer)
    self.assertEqual(good, compare)

  def test_beam(self):
    with Context(BEAM=1):
      ast = (Tensor.empty(N, N) @ Tensor.empty(N, N)).schedule_linear().src[-1].src[0]
    p = do_to_program(ast, self.renderer)
    good, compare, _ = replay_to_program(p, ast, self.renderer)
    self.assertEqual(good, compare)

if __name__ == '__main__':
  unittest.main(verbosity=2)
