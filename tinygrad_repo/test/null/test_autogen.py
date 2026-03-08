import ctypes, struct, subprocess, tempfile, unittest
from typing import Annotated
from tinygrad.helpers import OSX, WIN
from tinygrad.runtime.support.c import DLL, record, init_records
from tinygrad.runtime.support import c
from tinygrad.runtime.support.autogen import gen

@unittest.skipIf(WIN, "doesn't compile on windows")
class TestC(unittest.TestCase):
  def compile(self, src):
    with tempfile.NamedTemporaryFile(suffix=".so") as f:
      subprocess.check_output(('clang', '-x', 'c', '-fPIC', '-shared', '-', '-o', f.name), input=src.encode())
      return DLL("test", f.name)

  def test_packed_struct(self):
    @record
    class Baz:
      SIZE = 8
      a: Annotated[ctypes.c_uint, 0, 30]
      b: Annotated[ctypes.c_uint, 3, 30, 6]
      c: Annotated[ctypes.c_uint, 7, 2, 4]
      d: Annotated[ctypes.c_uint, 7, 2, 6]
    init_records()

    b = Baz(0x3AAADEAD, 0xBEEF, 1, 0)
    assert b.a == 0x3AAADEAD
    assert b.b == 0xBEEF
    assert b.c == 1
    assert b.d == 0

    b.a = 0xCAFE
    assert b.a == 0xCAFE
    assert b.b == 0xBEEF
    assert b.c == 1
    assert b.d == 0

  def test_packed_struct_interop(self):
    @record
    class Baz:
      SIZE = 8
      a: Annotated[ctypes.c_int, 0, 30]
      b: Annotated[ctypes.c_int, 3, 30, 6]
      c: Annotated[ctypes.c_int, 7, 2, 4]
      d: Annotated[ctypes.c_int, 7, 2, 6]
    init_records()

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
    dll = self.compile(src)
    b = Baz(0xAA000, 0x00BB0, 0, 1)
    @dll.bind
    def test(x:Baz) -> ctypes.c_int: ...
    self.assertEqual(test(b), b.a + b.b + b.c + b.d)

  # https://github.com/python/cpython/issues/90914
  def test_bitfield_interop(self):
    @record
    class Baz:
      SIZE = 1
      a: Annotated[ctypes.c_bool, 0, 1, 0]
      b: Annotated[ctypes.c_bool, 0, 1, 1]
      c: Annotated[ctypes.c_bool, 0, 1, 2]
      d: Annotated[ctypes.c_bool, 0, 1, 3]
      e: Annotated[ctypes.c_bool, 0, 1, 4]
      f: Annotated[ctypes.c_bool, 0, 1, 5]
      g: Annotated[ctypes.c_bool, 0, 1, 6]
      h: Annotated[ctypes.c_bool, 0, 1, 7]
    init_records()
    src = '''#include <stdbool.h>
      struct baz {
        bool a:1, b:1, c:1, d:1, e:1, f:1, g:1, h:1;
      };

      int test(struct baz x) {
        return x.c;
      }
    '''
    dll = self.compile(src)
    @dll.bind
    def test(x:Baz) -> ctypes.c_int: ...
    for i in range(8): self.assertEqual(test(Baz(*(j==i for j in range(8)))), i==2)

  def test_struct_interop(self):
    @record
    class Baz:
      SIZE = 32
      a: Annotated[ctypes.c_int, 0]
      b: Annotated[ctypes.c_int, 4]
      c: Annotated[ctypes.c_int, 8]
      d: Annotated[ctypes.c_int, 12]
      e: Annotated[ctypes.c_int, 16]
      f: Annotated[ctypes.c_int, 20]
      g: Annotated[ctypes.c_int, 24]
      h: Annotated[ctypes.c_int, 28]
    init_records()
    src = '''#include <stdio.h>
      struct baz {
        int a, b, c, d, e, f, g, h;
      };

      struct baz test(struct baz x) {
        return (struct baz){x.h, x.g, x.f, x.e, x.d, x.c, x.b, x.a};
      }
    '''
    dll = self.compile(src)
    @dll.bind
    def test(x:Baz) -> Baz: ...
    self.assertEqual(bytes(test(Baz(*range(8)))), struct.pack("8i", *range(7, -1, -1)))

  def test_aos_interop(self):
    @record
    class Item:
      SIZE = 4
      val: Annotated[ctypes.c_int, 0]
    init_records()
    src = """
    struct item { int val; };
      int test(struct item arr[3]) {
        int ret = 0;
        for (int i = 0; i < 3; i++) ret += arr[i].val;
        return ret;
      }
    """
    dll = self.compile(src)
    @dll.bind
    def test(arr:(Item * 3)) -> ctypes.c_int: ...
    self.assertEqual(test((Item * 3)(Item(10), Item(20), Item(30))), 60)

  def test_soa_interop(self):
    @record
    class Row:
      SIZE = 16
      data: Annotated[ctypes.c_int * 3, 0]
    init_records()
    src = """
      struct row { int data[3]; };
      struct row test(struct row x) {
        return (struct row){{ x.data[2], x.data[1], x.data[0] }};
      }
    """
    dll = self.compile(src)
    @dll.bind
    def test(x:Row) -> Row: ...
    r = test(Row((ctypes.c_int * 3)(10, 20, 30)))
    self.assertIsInstance(r, Row)
    self.assertEqual(r.data[0], 30)
    self.assertEqual(r.data[1], 20)
    self.assertEqual(r.data[2], 10)

  def test_soa_ptr_interop(self):
    @record
    class Row:
      SIZE = 8
      data: Annotated[c.POINTER[ctypes.c_int], 0]
    init_records()
    src = """
      struct row { int *data; };
      int test(struct row x) {
        return x.data[2] + x.data[1] + x.data[0];
      }
    """
    dll = self.compile(src)
    @dll.bind
    def test(x:Row) -> ctypes.c_int: ...
    assert test(Row((ctypes.c_int * 3)(10, 20, 30))) == 60

  def test_nested_struct_interop(self):
    @record
    class Inner:
      SIZE = 4
      a: Annotated[ctypes.c_int, 0]
    @record
    class Outer:
      SIZE = 8
      inner: Annotated[Inner, 0]
      b: Annotated[ctypes.c_int, 4]
    init_records()
    src = """
      struct i { int a; };
      struct o { struct i i; int b; };
      struct o test(struct o x) {
        return (struct o){(struct i){ x.b }, x.i.a };
      }
    """
    dll = self.compile(src)
    @dll.bind
    def test(x:Outer) -> Outer: ...
    o = test(Outer(Inner(10), 20))
    self.assertEqual(o.inner.a, 20)
    self.assertEqual(o.b, 10)

  def test_struct_pointer_interop(self):
    @record
    class Foo:
      SIZE = 8
      a: Annotated[ctypes.c_int, 0]
      b: Annotated[ctypes.c_int, 4]
    init_records()
    src = """
      struct foo { int a, b; };
      struct foo *test(struct foo *f) {
        int x = f->a;
        f->a = f->b;
        f->b = x;
        return f;
      }
    """
    dll = self.compile(src)
    @dll.bind
    def test(f:ctypes.POINTER(Foo)) -> ctypes.POINTER(Foo): ...
    inp = ctypes.pointer(Foo(10, 20))
    out = test(inp)
    self.assertEqual(out.contents.a, 20)
    self.assertEqual(out.contents.b, 10)

  def test_pointer_field_roundtrip(self):
    # This tests storing a pointer in a record struct field and passing it to C
    # Mimics how mesa.struct_lp_build_tgsi_params.mask is used
    from tinygrad.runtime.support.c import POINTER
    @record
    class Inner:
      SIZE = 8
      value: Annotated[ctypes.c_int, 0]
      flag: Annotated[ctypes.c_int, 4]
    @record
    class Outer:
      SIZE = 16
      x: Annotated[ctypes.c_int, 0]
      inner_ptr: Annotated[POINTER[Inner], 8]
    init_records()

    src = """
      struct inner { int value; int flag; };
      struct outer { int x; struct inner *inner_ptr; };
      int test(struct inner *p) {
        return p->value + p->flag;
      }
    """
    dll = self.compile(src)
    @dll.bind
    def test(p:POINTER[Inner]) -> ctypes.c_int: ...

    inner = Inner(value=42, flag=10)
    outer = Outer(x=1, inner_ptr=ctypes.pointer(inner))
    # Retrieve pointer from struct field and pass to C
    self.assertEqual(test(outer.inner_ptr), 52)

  def test_pointer_field_loses_reference(self):
    # BUG: When a pointer is stored in a record struct field, only the address bytes are saved.
    # The pointer's _objects dict (which prevents GC of the pointed-to object) is lost.
    # This causes the pointed-to object to be garbage collected, leading to use-after-free.
    from tinygrad.runtime.support.c import POINTER
    @record
    class MaskContext:
      SIZE = 16
      value: Annotated[ctypes.c_int, 0]
      initialized: Annotated[ctypes.c_int, 4]
      ptr: Annotated[ctypes.c_void_p, 8]
    @record
    class Params:
      SIZE = 16
      x: Annotated[ctypes.c_int, 0]
      mask: Annotated[POINTER[MaskContext], 8]
    init_records()

    src = """
      struct mask_ctx { int value; int initialized; void *ptr; };
      void mask_begin(struct mask_ctx *m, int val) { m->value = val; m->initialized = 1; }
      int mask_end(struct mask_ctx *m) { return m->value + m->initialized; }
    """
    dll = self.compile(src)
    @dll.bind
    def mask_begin(m:POINTER[MaskContext], val:ctypes.c_int) -> None: ...
    @dll.bind
    def mask_end(m:POINTER[MaskContext]) -> ctypes.c_int: ...

    # When MaskContext() is created inline, it gets garbage collected after the pointer
    # is stored because only the address bytes are saved, not the _objects reference.
    params = Params(x=1, mask=ctypes.pointer(MaskContext()))
    mask_begin(params.mask, 42)
    result = mask_end(params.mask)
    self.assertEqual(result, 43)  # 42 + 1

@unittest.skipIf(OSX and ('MTLCompiler' in DLL._loaded_ or 'llvm' in DLL._loaded_), "libclang can't be loaded after MTLCompiler or llvm on OSX")
@unittest.skipIf(WIN, "doesn't compile on windows")
class TestAutogen(unittest.TestCase):
  def run_gen(self, contents):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.h') as f:
      f.write(contents)
      f.flush()

      generated_code = gen(name="test_header", dll=None, files=[f.name])

      namespace = {}
      exec(generated_code, namespace)
      return namespace

  def test_packed_structs(self):
    ns = self.run_gen("""
typedef unsigned NvU32;
typedef unsigned long NvU64;

typedef struct
{
    NvU32 version;
    NvU32 size;
    NvU64 gfwImageOffset;
    NvU32 gfwImageSize;
    NvU32 flags;
} __attribute__((packed)) FWSECLIC_READ_VBIOS_DESC;

#define FWSECLIC_READ_VBIOS_STRUCT_FLAGS (2)

typedef struct
{
    NvU32 version;
    NvU32 size;
    NvU32 frtsRegionOffset4K;
    NvU32 frtsRegionSize;
    NvU32 frtsRegionMediaType;
} __attribute__((packed)) FWSECLIC_FRTS_REGION_DESC;

#define FWSECLIC_FRTS_REGION_MEDIA_FB (2)
#define FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K (0x100)

typedef struct
{
    FWSECLIC_READ_VBIOS_DESC readVbiosDesc;
    FWSECLIC_FRTS_REGION_DESC frtsRegionDesc;
} __attribute__((packed)) FWSECLIC_FRTS_CMD;
""")

    FWSECLIC_READ_VBIOS_DESC = ns['FWSECLIC_READ_VBIOS_DESC']
    FWSECLIC_FRTS_REGION_DESC = ns['FWSECLIC_FRTS_REGION_DESC']
    FWSECLIC_FRTS_CMD = ns['FWSECLIC_FRTS_CMD']

    read_vbios_desc = FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(FWSECLIC_READ_VBIOS_DESC), flags=2)
    frst_reg_desc = FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(FWSECLIC_FRTS_REGION_DESC),
      frtsRegionOffset4K=0xdead, frtsRegionSize=0x100, frtsRegionMediaType=2)
    frts_cmd = FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc)
    assert int.from_bytes(frts_cmd, 'little') == 0x2000001000000dead0000001400000001000000020000000000000000000000000000001800000001
    assert int.from_bytes(frts_cmd.readVbiosDesc, 'little') == int.from_bytes(read_vbios_desc, 'little')
    assert int.from_bytes(frts_cmd.frtsRegionDesc, 'little') == int.from_bytes(frst_reg_desc, 'little')
    assert frts_cmd.readVbiosDesc.__class__ is FWSECLIC_READ_VBIOS_DESC
    assert frts_cmd.frtsRegionDesc.__class__ is FWSECLIC_FRTS_REGION_DESC

  def test_gen_from_header(self):
    namespace = self.run_gen("""
    typedef struct {
      int x;
      int y;
    } Point;

    typedef enum {
      RED = 0,
      GREEN = 1,
      BLUE = 2
    } Color;

    typedef struct {
      Point origin;
      int width;
      int height;
      Color color;
    } Rectangle;

    int add_points(Point a, Point b);""")

    self.assertIn('Point', namespace)
    self.assertIn('Color', namespace)
    self.assertIn('Rectangle', namespace)
    self.assertIn('RED', namespace)
    self.assertIn('GREEN', namespace)
    self.assertIn('BLUE', namespace)

    self.assertEqual(namespace['RED'], 0)
    self.assertEqual(namespace['GREEN'], 1)
    self.assertEqual(namespace['BLUE'], 2)

    Point = namespace['Point']
    p = Point()
    self.assertTrue(hasattr(p, 'x'))
    self.assertTrue(hasattr(p, 'y'))

    Rectangle = namespace['Rectangle']
    rect = Rectangle()
    self.assertTrue(hasattr(rect, 'origin'))
    self.assertTrue(hasattr(rect, 'width'))
    self.assertTrue(hasattr(rect, 'height'))
    self.assertTrue(hasattr(rect, 'color'))

  def test_struct_ordering(self):
    namespace = self.run_gen("""
    struct A;
    struct C;
    typedef struct A A;

    struct B {
      struct C *c_ptr;
    };

    struct C {
      struct A *a_ptr;
    };

    struct A {
      int x;
      struct B *b_ptr;
    };""")

    self.assertIn('struct_A', namespace)
    self.assertIn('struct_B', namespace)
    self.assertIn('struct_C', namespace)
    A, B, C = namespace['A'], namespace['struct_B'], namespace['struct_C']
    a, b, c = A(), B(), C()
    self.assertTrue(hasattr(a, 'x'))
    self.assertTrue(hasattr(a, 'b_ptr'))
    self.assertTrue(hasattr(b, 'c_ptr'))
    self.assertTrue(hasattr(c, 'a_ptr'))

  def test_anonymous_children(self):
    namespace = self.run_gen("""
      struct foo {
        struct {
          int a,b;
        } bar;
      };
    """)

    self.assertIn('struct_foo', namespace)
    self.assertIn('struct_foo_bar', namespace)

  def test_enums(self):
    namespace = self.run_gen("""
      enum Foo { A, B, C };
      enum Bar { X, Y, Z };
    """)

    assert namespace["A"] == 0
    assert namespace["B"] == 1
    assert namespace["C"] == 2
    assert namespace["X"] == 0
    assert namespace["Y"] == 1
    assert namespace["Z"] == 2
    assert namespace["enum_Foo"].get(0) == "A"
    assert namespace["enum_Foo"].get(1) == "B"
    assert namespace["enum_Foo"].get(2) == "C"
    assert namespace["enum_Bar"].get(0) == "X"
    assert namespace["enum_Bar"].get(1) == "Y"
    assert namespace["enum_Bar"].get(2) == "Z"

  @unittest.skipIf(OSX, "can't find stdint?")
  def test_packed_fields(self):
    ns = self.run_gen("""#include <stdint.h>
typedef struct die_info
 {
	 uint16_t die_id;
	 uint16_t die_offset; /* Points to the corresponding die_header structure */
 } die_info;

typedef struct ip_discovery_header
 {
	 uint32_t signature;    /* Table Signature */
	 uint16_t version;      /* Table Version */
	 uint16_t size;         /* Table Size */
	 uint32_t id;           /* Table ID */
	 uint16_t num_dies;     /* Number of Dies */
	 die_info die_info[16]; /* list die information for up to 16 dies */
	 union {
		 uint16_t padding[1];	/* version <= 3 */
		 struct {		/* version == 4 */
			 uint8_t base_addr_64_bit : 1; /* ip structures are using 64 bit base address */
			 uint8_t reserved : 7;
			 uint8_t reserved2;
		 };
	 };
 } ip_discovery_header;
""")

    ip_discovery_header = ns['ip_discovery_header']

    hdr = b'IPDS\x04\x00|\x1d\x80\x1a\xffd\x01\x00\x00\x00\x8c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00' # noqa: E501
    ihdr = ip_discovery_header.from_buffer_copy(hdr)

    assert ctypes.sizeof(ihdr) == 80
    assert ihdr.signature == 0x53445049
    assert ihdr.version == 0x0004
    assert ihdr.num_dies == 1
    assert ihdr.base_addr_64_bit == 1

if __name__ == "__main__": unittest.main()
