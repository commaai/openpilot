import unittest, io
from contextlib import redirect_stdout
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import OSX
from tinygrad.engine.realize import compile_linear
from tinygrad.codegen import to_program

class TestCompileFailures(unittest.TestCase):
  def compile(self, out:Tensor):
    compile_linear(out.schedule_linear())

  def test_interpolate_atari(self):
    self.compile(Tensor.empty(210, 160, dtype='uint8').interpolate((64, 64)))

  def test_add_max_uchar(self):
    self.compile((Tensor.empty(1024, dtype='uint8') + Tensor.empty(1024, dtype='uint8')).max())

class TestDisassembly(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "CPU" and OSX, "m series cpus support fp16 arithmetic")
  def test_float16_alu(self):
    c = Tensor([1], dtype=dtypes.float16) + Tensor([1], dtype=dtypes.float16)
    s = c.schedule_linear().src[-1]
    p = to_program(s.src[0], Device[Device.DEFAULT].renderer)
    lib = Device[Device.DEFAULT].compiler.compile(p.src[2].arg)
    out = io.StringIO()
    with redirect_stdout(out): Device[Device.DEFAULT].compiler.disassemble(lib)
    assert "fcvt" not in out.getvalue()

if __name__ == '__main__':
  unittest.main()
