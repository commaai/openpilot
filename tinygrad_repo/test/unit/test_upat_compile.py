import unittest
from tinygrad.helpers import DEBUG
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UPat, track_rewrites, GroupOp, Ops
from tinygrad.uop.upat import _get_code, upat_compile
import dis

@track_rewrites()
def do_compile(up):
  print("\n***** COMPILE", up)
  match_code = _get_code(up, False)
  match = upat_compile(up, lambda **kwargs: None)
  print(match_code[0])
  if DEBUG >= 2: dis.dis(match)
  return match_code[0]

class TestUPatCompile(unittest.TestCase):
  def test_double(self):
    up = UPat.var("x") * UPat.cvar("c0") + UPat.var("x") * UPat.cvar("c1")
    do_compile(up)

  def test_single(self):
    up = UPat.var("x") + UPat.var("y")
    do_compile(up)

  def test_xpx(self):
    up = UPat.var("x") + UPat.var("x")
    do_compile(up)

  def test_xp0(self):
    up = UPat.var("x") + 0
    do_compile(up)

  def test_bool(self):
    up = UPat.var('x', dtype=dtypes.bool) * UPat.var('y', dtype=dtypes.bool)
    do_compile(up)

  def test_single_c(self):
    up = (UPat.var("x") + UPat.var("y")) * UPat.var("c")
    do_compile(up)

  def test_const_folding(self):
    up = UPat(GroupOp.ALU-{Ops.THREEFRY}, name="a", src=UPat((Ops.VCONST, Ops.CONST)))
    do_compile(up)

  @unittest.skip("fix this")
  def test_range_named(self):
    # this should be one src, but this should also still work
    up = UPat(Ops.CAST, dtypes.float, UPat.var("x", dtypes.bfloat16))
    do_compile(up)

if __name__ == "__main__":
  unittest.main()
