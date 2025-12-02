import ctypes, subprocess, tempfile, unittest
from tinygrad.helpers import WIN
from tinygrad.runtime.support.c import Struct

class TestAutogen(unittest.TestCase):
  def test_packed_struct_sizeof(self):
    layout = [('a', ctypes.c_char), ('b', ctypes.c_int, 5), ('c', ctypes.c_char)]
    class X(ctypes.Structure): _fields_, _layout_ = layout, 'gcc-sysv'
    class Y(ctypes.Structure): _fields_, _pack_, _layout_ = layout, 1, 'ms'
    class Z(Struct): _packed_, _fields_ = True, layout
    self.assertNotEqual(ctypes.sizeof(X), 4) # ctypes bug! gcc-13.3.0 says this should have size 4
    self.assertEqual(ctypes.sizeof(Y), 6)
    self.assertEqual(ctypes.sizeof(Z), 3)
    layout = [('a', ctypes.c_int, 31), ('b', ctypes.c_int, 31), ('c', ctypes.c_int, 1), ('d', ctypes.c_int, 1)]
    class Foo(ctypes.Structure): _fields_, _layout_ = layout, 'gcc-sysv'
    class Bar(ctypes.Structure): _fields_, _pack_, _layout_ = layout, 1, 'ms'
    class Baz(Struct): _fields_, _packed_ = layout, True
    self.assertEqual(ctypes.sizeof(Foo), 12)
    self.assertEqual(ctypes.sizeof(Bar), 12)
    self.assertEqual(ctypes.sizeof(Baz), 8)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_packed_struct_interop(self):
    class Baz(Struct): pass
    Baz._packed_ = True
    Baz._fields_ = [('a', ctypes.c_int, 30), ('b', ctypes.c_int, 30), ('c', ctypes.c_int, 2), ('d', ctypes.c_int, 2)]
    src = '''
      struct __attribute__((packed)) baz {
        int a:30;
        int b:30;
        int c:2;
        int d:2;
      };

      int test(struct baz x) {
        return x.a + x.b + x.c + x.d;
      }
    '''
    args = ('-x', 'c', '-fPIC', '-shared')
    with tempfile.NamedTemporaryFile(suffix=".so") as f:
      subprocess.check_output(('clang',) + args + ('-', '-o', f.name), input=src.encode('utf-8'))
      b = Baz(0xAA000, 0x00BB0, 0, 1)
      test = ctypes.CDLL(f.name).test
      test.argtypes = [Baz]
      self.assertEqual(test(b), b.a + b.b + b.c + b.d)

if __name__ == "__main__": unittest.main()
