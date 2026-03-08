"""Tests comparing sqtt.py PACKET_TYPES_L3/L4 against AMD's rocprof-trace-decoder binary."""
import unittest, struct, ctypes, pickle
from pathlib import Path

ROCPROF_LIB = Path("/usr/lib/librocprof-trace-decoder.so")
EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "sqtt/examples"

def _find_segment(perms: str):
  """Find a segment of the loaded library with given permissions (e.g. 'rw-p', 'r--p')."""
  with open('/proc/self/maps', 'r') as f:
    for line in f:
      if 'librocprof-trace-decoder.so' in line and f' {perms} ' in line:
        parts = line.split()
        return int(parts[0].split('-')[0], 16), int(parts[2], 16)
  return None, None

def _read_array(file_offset: int, count: int):
  """Read an array of uint8 at file_offset from the loaded library."""
  base, seg_offset = _find_segment('rw-p')
  if base is None: return None
  return list((ctypes.c_uint8 * count).from_address(base + (file_offset - seg_offset)))

def _load_lib():
  if not ROCPROF_LIB.exists(): return False
  ctypes.CDLL(str(ROCPROF_LIB))
  return True

# ═══════════════════════════════════════════════════════════════════════════════
# RDNA EXTRACTION (nibble-based format)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_bit_tables():
  """Extract bit budget tables. Returns (layout2, layout3, layout4) or None."""
  if not _load_lib(): return None
  return _read_array(0x2d220, 32), _read_array(0x2d280, 32), _read_array(0x2d2c0, 32)

def extract_delta_fields():
  """Extract delta bitfield tables. Returns (layout2, layout3, layout4) dicts mapping type_id -> (lo, hi)."""
  if not _load_lib(): return None
  ro_base, ro_offset = _find_segment('r--p')
  if ro_base is None: return None

  def read_table(file_offset, num_entries):
    addr = ro_base + (file_offset - ro_offset)
    data = bytes((ctypes.c_uint8 * (num_entries * 12)).from_address(addr))
    return {type_id: (lo, hi) for j in range(0, len(data), 12)
            for type_id, lo, hi in [struct.unpack('<III', data[j:j+12])] if type_id < 32}

  return read_table(0x26800, 24), read_table(0x26dc0, 25), read_table(0x27300, 27)

def extract_packet_encodings():
  """Extract packet encodings. Returns (L2, L3, L4) dicts mapping type_id -> (mask, value)."""
  if not _load_lib(): return None
  rw_base, rw_offset = _find_segment('rw-p')
  if rw_base is None: return None

  # Read base encodings from registration vector at 0x2d340
  vec_start = ctypes.c_void_p.from_address(rw_base + (0x2d340 - rw_offset)).value
  vec_end = ctypes.c_void_p.from_address(rw_base + (0x2d348 - rw_offset)).value
  base = {}
  if vec_start and vec_end:
    for i in range((vec_end - vec_start) // 32):
      addr = vec_start + i * 32
      type_id = ctypes.c_uint8.from_address(addr).value
      pat_start = ctypes.c_void_p.from_address(addr + 8).value
      pat_end = ctypes.c_void_p.from_address(addr + 16).value
      if pat_start and pat_end and 0 < (n := pat_end - pat_start) <= 8:
        pat = list((ctypes.c_uint8 * n).from_address(pat_start))
        base[type_id] = (sum(1 << j for j in range(n)), sum(b << j for j, b in enumerate(pat)))

  return {**base, 17: (0x7f, 0x51), 25: (0x7f, 0x31)}, base, {**base}  # L2 has overrides

# ═══════════════════════════════════════════════════════════════════════════════
# CDNA EXTRACTION (16-bit header format)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_cdna_packet_sizes():
  """Extract CDNA pkt_fmt -> size mapping by running rocprof decoder to populate its hash table."""
  from extra.assembly.amd.test.test_sqtt_examples import run_rocprof_decoder

  if not (pkl_path := next((EXAMPLES_DIR / "gfx950").glob("*.pkl"), None)): return None
  with open(pkl_path, "rb") as f: data = pickle.load(f)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  prg = next((e for e in data if type(e).__name__ == "ProfileProgramEvent"), None)
  if not sqtt_events or not prg: return None

  # Run decoder to trigger hash table initialization
  run_rocprof_decoder([e.blob for e in sqtt_events], prg.lib, prg.base, "gfx950")

  # Extract hash table: head at 0x2d4f0, nodes are 16 bytes (next[8], key[4], value[4])
  rw_base, rw_offset = _find_segment('rw-p')
  if not (head := ctypes.c_void_p.from_address(rw_base + (0x2d4f0 - rw_offset)).value if rw_base else None): return None

  pkt_sizes, node, seen = {}, head, set()
  while node and node not in seen and len(pkt_sizes) < 20:
    seen.add(node)
    key, val = ctypes.c_uint32.from_address(node + 8).value, ctypes.c_uint32.from_address(node + 12).value
    if key < 16 and val in (0x10, 0x20, 0x30, 0x40): pkt_sizes[key] = {0x10: 2, 0x20: 4, 0x30: 6, 0x40: 8}[val]
    node = ctypes.c_void_p.from_address(node).value
  return pkt_sizes if len(pkt_sizes) == 16 else None

# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSQTTMatchesBinary(unittest.TestCase):
  def test_bit_counts_match_layout3(self): self._test_bit_counts(3)
  def test_bit_counts_match_layout4(self): self._test_bit_counts(4)
  def test_encodings_match_layout3(self): self._test_encodings(3)
  def test_encodings_match_layout4(self): self._test_encodings(4)
  def test_delta_fields_match_layout3(self): self._test_delta_fields(3)
  def test_delta_fields_match_layout4(self): self._test_delta_fields(4)

  def test_cdna_packet_sizes(self):
    """Extract and verify CDNA pkt_fmt -> size mapping from rocprof's hash table."""
    if not (EXAMPLES_DIR / "gfx950").exists(): self.skipTest("no CDNA examples")
    pkt_sizes = extract_cdna_packet_sizes()
    self.assertIsNotNone(pkt_sizes, "failed to extract CDNA packet sizes")
    from extra.assembly.amd.sqtt_cdna import CDNA_PKT_SIZES
    for pkt_fmt, size in CDNA_PKT_SIZES.items():
      with self.subTest(pkt_fmt=pkt_fmt): self.assertEqual(pkt_sizes.get(pkt_fmt), size)

  def _test_bit_counts(self, layout: int):
    if not (tables := extract_bit_tables()): self.skipTest("rocprof-trace-decoder not installed")
    from extra.assembly.amd.sqtt import PACKET_TYPES_L3, PACKET_TYPES_L4
    for type_id, pkt_cls in {3: PACKET_TYPES_L3, 4: PACKET_TYPES_L4}[layout].items():
      with self.subTest(packet=pkt_cls.__name__):
        self.assertEqual(pkt_cls._size_nibbles * 4, tables[layout - 2][type_id])

  def _test_encodings(self, layout: int):
    if not (encodings := extract_packet_encodings()): self.skipTest("rocprof-trace-decoder not installed")
    from extra.assembly.amd.sqtt import PACKET_TYPES_L3, PACKET_TYPES_L4
    for type_id, pkt_cls in {3: PACKET_TYPES_L3, 4: PACKET_TYPES_L4}[layout].items():
      with self.subTest(packet=pkt_cls.__name__):
        self.assertEqual((pkt_cls.encoding.mask, pkt_cls.encoding.default), encodings[layout - 2][type_id])

  def _test_delta_fields(self, layout: int):
    if not (deltas := extract_delta_fields()): self.skipTest("rocprof-trace-decoder not installed")
    from extra.assembly.amd.sqtt import PACKET_TYPES_L3, PACKET_TYPES_L4
    for type_id, pkt_cls in {3: PACKET_TYPES_L3, 4: PACKET_TYPES_L4}[layout].items():
      if type_id not in deltas[layout - 2]: continue
      delta = getattr(pkt_cls, 'delta', None)
      actual = (0, 0) if delta is None else (delta.lo, delta.hi + 1)
      with self.subTest(packet=pkt_cls.__name__): self.assertEqual(actual, deltas[layout - 2][type_id])

if __name__ == "__main__":
  tables = extract_bit_tables()
  encodings = extract_packet_encodings()
  deltas = extract_delta_fields()

  TYPE_NAMES = {1: 'VALUINST', 2: 'VMEMEXEC', 3: 'ALUEXEC', 4: 'IMMEDIATE', 5: 'IMMEDIATE_MASK', 6: 'WAVERDY',
    7: 'TS_DELTA_S8_W3', 8: 'WAVEEND', 9: 'WAVESTART', 10: 'TS_DELTA_S5_W2', 11: 'WAVEALLOC', 12: 'TS_DELTA_S5_W3',
    13: 'PERF', 14: 'UTILCTR', 15: 'TS_DELTA_SHORT', 16: 'NOP', 17: 'TS_WAVE_STATE', 18: 'EVENT', 19: 'EVENT_BIG',
    20: 'REG', 21: 'SNAPSHOT', 22: 'TS_DELTA_OR_MARK', 23: 'LAYOUT_HEADER', 24: 'INST', 25: 'UNK_25'}

  print("L2:", tables[0], "\nL3:", tables[1], "\nL4:", tables[2])
  if encodings and tables:
    print(f"\n{'TypeID':>6} {'Name':>18} {'L2 enc':>12} {'L3 enc':>12} {'L4 enc':>12} {'L2':>4} {'L3':>4} {'L4':>4} {'L2 delta':>12} {'L3 delta':>12} {'L4 delta':>12}")
    print("-" * 140)
    for type_id in sorted(set(encodings[0]) | set(encodings[1]) | set(encodings[2])):
      name = TYPE_NAMES.get(type_id, f'UNK_{type_id}')
      bits = [tables[i][type_id] if type_id < len(tables[i]) else 0 for i in range(3)]
      enc_strs = [f"0x{encodings[i][type_id][0]:02x}/0x{encodings[i][type_id][1]:02x}" if type_id in encodings[i] else "-" for i in range(3)]
      delta_strs = [f"[{d[1]-1}:{d[0]}]" if (d := deltas[i].get(type_id, (0, 0)))[1] > d[0] else "-" for i in range(3)]
      print(f"{type_id:6d} {name:>18} {enc_strs[0]:>12} {enc_strs[1]:>12} {enc_strs[2]:>12} {bits[0]:4d} {bits[1]:4d} {bits[2]:4d} {delta_strs[0]:>12} {delta_strs[1]:>12} {delta_strs[2]:>12}")

  cdna = extract_cdna_packet_sizes()
  if cdna: print(f"\nCDNA packet sizes: {cdna}")

  unittest.main()
