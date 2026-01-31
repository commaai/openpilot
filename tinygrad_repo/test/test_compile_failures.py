import unittest, io
from contextlib import redirect_stdout
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import OSX, CPU_LLVM, CPU_LVP
from tinygrad.device import is_dtype_supported
from tinygrad.engine.realize import get_program

class TestCompileFailures(unittest.TestCase):
  def compile(self, out:Tensor):
    for si in out.schedule(): si.lower()

  @unittest.skipUnless(is_dtype_supported(dtypes.uchar, Device.DEFAULT), f"no uint8 on {Device.DEFAULT}")
  def test_interpolate_atari(self):
    self.compile(Tensor.empty(210, 160, dtype='uint8').interpolate((64, 64)))

  def test_add_max_uchar(self):
    self.compile((Tensor.empty(1024, dtype='uint8') + Tensor.empty(1024, dtype='uint8')).max())

class TestDisassembly(unittest.TestCase):
  # TODO: fails on llvm. llvm.LLVMGetHostCPUName() returns "generic"
  @unittest.skipUnless(Device.DEFAULT in ("CPU",) and not (CPU_LLVM or CPU_LVP) and OSX, "m series cpus support fp16 arithmetic")
  def test_float16_alu(self):
    c = Tensor([1], dtype=dtypes.float16) + Tensor([1], dtype=dtypes.float16)
    s = c.schedule()[-1]
    p = get_program(s.ast, Device[Device.DEFAULT].renderer)
    lib = Device[Device.DEFAULT].compiler.compile(p.src)
    out = io.StringIO()
    with redirect_stdout(out): Device[Device.DEFAULT].compiler.disassemble(lib)
    assert "fcvt" not in out.getvalue()

if __name__ == '__main__':
  unittest.main()
