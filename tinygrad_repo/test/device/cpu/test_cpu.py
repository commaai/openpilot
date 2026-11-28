import unittest, io
from contextlib import redirect_stdout
from tinygrad import Tensor, Device
from tinygrad.helpers import DEV, Target
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.x86 import X86Renderer
from tinygrad.renderer.cstyle import TCCRenderer
from tinygrad.codegen import to_program
if DEV.renderer == "LLVM": from test.runtime import test_ops

@unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
class TestCPU(unittest.TestCase):
  def test_arch_feats(self):
    ast = (Tensor.empty(16) + Tensor.empty(16)).schedule_linear().src[-1].src[0]
    for ren in Device[Device.DEFAULT].renderers:
      for arch, expect_vmov in [("x86_64,x86-64,avx", True), ("x86_64,x86-64,-avx", False)]:
        with self.subTest(arch=arch):
          if ren is X86Renderer: continue # X86 requires avx support
          if ren in (LVPRenderer, TCCRenderer): continue # LVP and TCC do not play nice with cross compilation
          r = ren(Target(device="CPU", arch=arch))
          p = to_program(ast, r)
          lib = r.compiler.compile(p.src[2].arg)
          out = io.StringIO()
          with redirect_stdout(out): r.compiler.disassemble(lib)
          self.assertEqual("vmov" in out.getvalue(), expect_vmov, out.getvalue())

class TestOps(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "CPU" and DEV.renderer == "LLVM", "DEVECTORIZE=0 only for LLVM")
  def test_strided_conv2d_simple_vec(self):
    test_ops.TestOps.test_strided_conv2d_simple(self)

if __name__ == '__main__':
  unittest.main()
