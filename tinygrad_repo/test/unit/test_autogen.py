import ctypes, subprocess, tempfile, unittest
from tinygrad.helpers import WIN
from tinygrad.runtime.support.c import Struct

class TestAutogen(unittest.TestCase):
  def test_packed_struct_sizeof(self):
    layout = [('a', ctypes.c_char), ('b', ctypes.c_int, 5), ('c', ctypes.c_char)]
    class Y(ctypes.Structure): _fields_, _pack_, _layout_ = layout, 1, 'ms'
    class Z(Struct): pass
    Z._packed_, Z._fields_ = True, layout
    self.assertEqual(ctypes.sizeof(Y), 6)
    self.assertEqual(ctypes.sizeof(Z), 3)
    layout = [('a', ctypes.c_int, 31), ('b', ctypes.c_int, 31), ('c', ctypes.c_int, 1), ('d', ctypes.c_int, 1)]
    class Foo(ctypes.Structure): _fields_, _layout_ = layout, 'gcc-sysv'
    class Bar(ctypes.Structure): _fields_, _pack_, _layout_ = layout, 1, 'ms'
    class Baz(Struct): pass
    Baz._packed_, Baz._fields_ = True, layout
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

  # https://github.com/python/cpython/issues/90914
  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_bitfield_interop(self):
    class Baz(Struct): pass
    Baz._fields_ = [(chr(ord('a') + i), ctypes.c_bool, 1) for i in range(8)]
    src = '''#include <stdbool.h>
      struct baz {
        bool a:1;
        bool b:1;
        bool c:1;
        bool d:1;
        bool e:1;
        bool f:1;
        bool g:1;
        bool h:1;
      };

      int test(struct baz x) {
        return x.c;
      }
    '''
    args = ('-x', 'c', '-fPIC', '-shared')
    with tempfile.NamedTemporaryFile(suffix=".so") as f:
      subprocess.check_output(('clang',) + args + ('-', '-o', f.name), input=src.encode('utf-8'))
      test = ctypes.CDLL(f.name).test
      test.argtypes = [Baz]
      for i in range(8): self.assertEqual(test(Baz(*(j==i for j in range(8)))), i==2)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_packed_structs(self):
    NvU32 = ctypes.c_uint32
    NvU64 = ctypes.c_uint64
    class FWSECLIC_READ_VBIOS_DESC(Struct): pass
    FWSECLIC_READ_VBIOS_DESC._packed_ = True
    FWSECLIC_READ_VBIOS_DESC._fields_ = [
      ('version', NvU32),
      ('size', NvU32),
      ('gfwImageOffset', NvU64),
      ('gfwImageSize', NvU32),
      ('flags', NvU32),
    ]
    class FWSECLIC_FRTS_REGION_DESC(Struct): pass
    FWSECLIC_FRTS_REGION_DESC._packed_ = True
    FWSECLIC_FRTS_REGION_DESC._fields_ = [
      ('version', NvU32),
      ('size', NvU32),
      ('frtsRegionOffset4K', NvU32),
      ('frtsRegionSize', NvU32),
      ('frtsRegionMediaType', NvU32),
    ]
    class FWSECLIC_FRTS_CMD(Struct): pass
    FWSECLIC_FRTS_CMD._packed_ = True
    FWSECLIC_FRTS_CMD._fields_ = [
      ('readVbiosDesc', FWSECLIC_READ_VBIOS_DESC),
      ('frtsRegionDesc', FWSECLIC_FRTS_REGION_DESC),
    ]
    read_vbios_desc = FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(FWSECLIC_READ_VBIOS_DESC), flags=2)
    frst_reg_desc = FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(FWSECLIC_FRTS_REGION_DESC),
      frtsRegionOffset4K=0xdead, frtsRegionSize=0x100, frtsRegionMediaType=2)
    frts_cmd = FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc)
    assert int.from_bytes(frts_cmd, 'little') == 0x2000001000000dead0000001400000001000000020000000000000000000000000000001800000001
    assert int.from_bytes(frts_cmd.readVbiosDesc, 'little') == int.from_bytes(read_vbios_desc, 'little')
    assert int.from_bytes(frts_cmd.frtsRegionDesc, 'little') == int.from_bytes(frst_reg_desc, 'little')
    assert frts_cmd.readVbiosDesc.__class__ is FWSECLIC_READ_VBIOS_DESC
    assert frts_cmd.frtsRegionDesc.__class__ is FWSECLIC_FRTS_REGION_DESC

  def test_packed_fields(self):
    uint8_t = ctypes.c_ubyte
    uint16_t = ctypes.c_ushort
    uint32_t = ctypes.c_uint32

    class struct_die_info(Struct): pass
    struct_die_info._packed_ = True
    struct_die_info._fields_ = [
      ('die_id', uint16_t),
      ('die_offset', uint16_t),
    ]
    die_info = struct_die_info
    class struct_ip_discovery_header(Struct): pass
    class struct_ip_discovery_header_0(ctypes.Union): pass
    class struct_ip_discovery_header_0_0(Struct): pass
    uint8_t = ctypes.c_ubyte
    struct_ip_discovery_header_0_0._fields_ = [
      ('base_addr_64_bit', uint8_t,1),
      ('reserved', uint8_t,7),
      ('reserved2', uint8_t),
    ]
    struct_ip_discovery_header_0._anonymous_ = ['_0']
    struct_ip_discovery_header_0._packed_ = True
    struct_ip_discovery_header_0._fields_ = [
      ('padding', (uint16_t * 1)),
      ('_0', struct_ip_discovery_header_0_0),
    ]
    struct_ip_discovery_header._anonymous_ = ['_0']
    struct_ip_discovery_header._packed_ = True
    struct_ip_discovery_header._fields_ = [
      ('signature', uint32_t),
      ('version', uint16_t),
      ('size', uint16_t),
      ('id', uint32_t),
      ('num_dies', uint16_t),
      ('die_info', (die_info * 16)),
      ('_0', struct_ip_discovery_header_0),
    ]
    ip_discovery_header = struct_ip_discovery_header

    hdr = b'IPDS\x04\x00|\x1d\x80\x1a\xffd\x01\x00\x00\x00\x8c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00' # noqa: E501
    ihdr = ip_discovery_header.from_buffer_copy(hdr)

    assert ctypes.sizeof(ihdr) == 80
    assert ihdr.signature == 0x53445049
    assert ihdr.version == 0x0004
    assert ihdr.num_dies == 1
    assert ihdr.base_addr_64_bit == 1

if __name__ == "__main__": unittest.main()
