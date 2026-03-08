#!/usr/bin/env python3
"""Tests for SQTT packet decoding using real captured examples."""
import pickle, unittest, ctypes, threading
from pathlib import Path
from tinygrad.helpers import DEBUG
from tinygrad.runtime.autogen import rocprof
from tinygrad.runtime.support.elf import elf_loader
from extra.assembly.amd import decode_inst
from extra.assembly.amd.autogen.rdna3.ins import SOPP
from extra.assembly.amd.autogen.rdna3.enum import SOPPOp
from extra.assembly.amd.sqtt import (decode, LAYOUT_HEADER, WAVESTART, WAVESTART_L4, WAVEEND, INST, INST_L4, VALUINST, IMMEDIATE, IMMEDIATE_MASK,
                                     ALUEXEC, VMEMEXEC, PACKET_TYPES_L3, PACKET_TYPES_L4, InstOp, InstOpL4, print_packets)
from extra.assembly.amd.test.helpers import TARGET_TO_ARCH

EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "sqtt/examples"
# INST ops for non-traced SIMDs (excluded from instruction count)
OTHER_SIMD_OPS = {InstOp.OTHER_LDS_LOAD, InstOp.OTHER_LDS_STORE, InstOp.OTHER_LDS_STORE_64, InstOp.OTHER_LDS_STORE_128,
                  InstOp.OTHER_FLAT_LOAD, InstOp.OTHER_FLAT_STORE, InstOp.OTHER_FLAT_STORE_64, InstOp.OTHER_FLAT_STORE_96,
                  InstOp.OTHER_FLAT_STORE_128, InstOp.OTHER_GLOBAL_LOAD, InstOp.OTHER_GLOBAL_LOAD_VADDR,
                  InstOp.OTHER_GLOBAL_STORE_64, InstOp.OTHER_GLOBAL_STORE_96, InstOp.OTHER_GLOBAL_STORE_128,
                  InstOp.OTHER_GLOBAL_STORE_VADDR_128}
OTHER_SIMD_OPS_L4 = {InstOpL4.OTHER_VMEM, InstOpL4.UNK_60}

# ═══════════════════════════════════════════════════════════════════════════════
# ROCPROF DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def run_rocprof_decoder(blobs: list[bytes], lib: bytes, base: int, target: str):
  """Run rocprof decoder on SQTT blobs, returning raw occupancy and instruction records."""
  image, sections, _ = elf_loader(lib)
  text = next((sh for sh in sections if sh.name == ".text"), None)
  assert text is not None, "no .text section found"
  text_off, text_size = text.header.sh_addr, text.header.sh_size

  blob_iter, current_blob = iter(blobs), [None]
  occupancy_records: list[tuple[int, int, int, int, bool]] = []  # (wave_id, simd, cu, time, is_start)
  wave_insts: list[list[tuple[int, int]]] = []  # per-wave list of (time, stall)

  @rocprof.rocprof_trace_decoder_se_data_callback_t
  def copy_cb(buf, buf_size, _):
    blob = next(blob_iter, None)
    if blob is None: return 0
    current_blob[0] = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
    buf[0] = ctypes.cast(current_blob[0], ctypes.POINTER(ctypes.c_ubyte))
    buf_size[0] = len(current_blob[0])
    return len(current_blob[0])

  @rocprof.rocprof_trace_decoder_trace_callback_t
  def trace_cb(record_type, events_ptr, n, _):
    if record_type == rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
      for ev in (rocprof.rocprofiler_thread_trace_decoder_occupancy_t * n).from_address(events_ptr):
        occupancy_records.append((ev.wave_id, ev.simd, ev.cu, ev.time, ev.start))
    elif record_type == rocprof.ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
      for ev in (rocprof.rocprofiler_thread_trace_decoder_wave_t * n).from_address(events_ptr):
        if ev.instructions_size > 0:
          sz = ev.instructions_size * ctypes.sizeof(rocprof.rocprofiler_thread_trace_decoder_inst_t)
          insts_blob = bytearray(sz)
          ctypes.memmove((ctypes.c_char * sz).from_buffer(insts_blob), ev.instructions_array, sz)
          insts = list((rocprof.rocprofiler_thread_trace_decoder_inst_t * ev.instructions_size).from_buffer(insts_blob))
          wave_insts.append([(inst.time, inst.stall) for inst in insts])
    return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

  arch = TARGET_TO_ARCH[target]
  @rocprof.rocprof_trace_decoder_isa_callback_t
  def isa_cb(instr_ptr, mem_size_ptr, size_ptr, pc, _):
    offset = pc.address - base
    if offset < text_off or offset >= text_off + text_size:
      mem_size_ptr[0] = 0
      return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    try:
      inst = decode_inst(image[offset:], arch=arch)
      mem_size_ptr[0] = inst._size()
    # this could be an error in our decode_inst
    except (ValueError, AssertionError):
      mem_size_ptr[0] = 0
      return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    if isinstance(inst, SOPP) and inst.op == SOPPOp.S_ENDPGM: mem_size_ptr[0] = 0
    # rocprof parses instruction string to determine type; v_nop works for all
    if (max_sz := size_ptr[0]) == 0: return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES
    ctypes.memmove(instr_ptr, b"v_nop", min(5, max_sz - 1))
    size_ptr[0] = min(5, max_sz - 1)
    return rocprof.ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS

  exc = None
  def worker():
    nonlocal exc
    try: rocprof.rocprof_trace_decoder_parse_data(copy_cb, trace_cb, isa_cb, None)
    except Exception as e: exc = e
  (t:=threading.Thread(target=worker, daemon=True)).start()
  t.join(timeout=1)
  if exc is not None: raise exc
  if t.is_alive(): raise RuntimeError("rocprof decoder timeout")
  return occupancy_records, wave_insts

class SQTTExamplesTestBase(unittest.TestCase):
  target: str

  @classmethod
  def setUpClass(cls):
    if cls is SQTTExamplesTestBase: raise unittest.SkipTest("base class")
    cls.examples = {}
    for pkl_path in sorted((EXAMPLES_DIR/cls.target).glob("*.pkl")):
      with open(pkl_path, "rb") as f:
        data = pickle.load(f)
      sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
      prg = next((e for e in data if type(e).__name__ == "ProfileProgramEvent"), None)
      if sqtt_events and prg:
        cls.examples[pkl_path.stem] = (sqtt_events, prg.lib, prg.base)

  def test_examples_loaded(self):
    self.assertGreater(len(self.examples), 0, "no example files found")

  def test_decode_all_examples(self):
    for name, (events, *_) in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          packets = list(decode(event.blob))
          if DEBUG >= 2: print(f"\n=== {name} event {i} ==="); print_packets(packets)
          self.assertGreater(len(packets), 0, f"no packets decoded from {name} event {i}")
          self.assertIsInstance(packets[0], LAYOUT_HEADER, f"first packet should be LAYOUT_HEADER in {name}")

  def test_packet_types_valid(self):
    all_classes = set(PACKET_TYPES_L3.values()) | set(PACKET_TYPES_L4.values())
    for name, (events, *_) in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          for pkt in decode(event.blob):
            # Use isinstance to handle layout-specific subclasses (e.g., WAVESTART_L4)
            self.assertTrue(any(isinstance(pkt, cls) for cls in all_classes), f"unknown packet type {type(pkt)} in {name}")

  def test_wave_lifecycle(self):
    for name, (events, *_) in self.examples.items():
      if "empty" in name: continue
      with self.subTest(example=name):
        all_packets = [p for e in events for p in decode(e.blob)]
        self.assertGreater(len([p for p in all_packets if isinstance(p, (WAVESTART, WAVESTART_L4))]), 0, f"no WAVESTART in {name}")
        self.assertGreater(len([p for p in all_packets if isinstance(p, WAVEEND)]), 0, f"no WAVEEND in {name}")

  def test_time_monotonic(self):
    for name, (events, *_) in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          times = [p._time for p in decode(event.blob)]
          self.assertEqual(times, sorted(times), f"timestamps not monotonic in {name}")

  def test_gemm_has_instructions(self):
    for name, (events, *_) in self.examples.items():
      if "gemm" not in name: continue
      with self.subTest(example=name):
        all_packets = [p for e in events for p in decode(e.blob)]
        self.assertGreater(len([p for p in all_packets if isinstance(p, (INST, INST_L4))]), 0, f"no INST packets in {name}")

  expected: dict[str, list[int]] = {}  # override in subclasses
  def test_packet_counts(self):
    if not self.expected: self.skipTest("no expected packet counts for this target")
    for name, (events, *_) in self.examples.items():
      with self.subTest(example=name):
        if not self.expected.get(name): continue
        counts = [len(list(decode(e.blob))) for e in events]
        self.assertEqual(counts, self.expected[name], f"packet count mismatch in {name}")

  def test_rocprof_wave_times_match(self):
    """Wave start/end times must match rocprof exactly."""
    for name, (events, lib, base) in self.examples.items():
      with self.subTest(example=name):
        occupancy, _ = run_rocprof_decoder([e.blob for e in events], lib, base, self.target)
        # extract from rocprof occupancy records
        roc_starts: dict[tuple[int, int, int], int] = {}
        roc_waves: list[tuple[int, int]] = []
        for wave_id, simd, cu, time, is_start in occupancy:
          key = (wave_id, simd, cu)
          if is_start: roc_starts[key] = time
          elif key in roc_starts: roc_waves.append((roc_starts.pop(key), time))
        # extract from our decoder
        our_waves: list[tuple[int, int]] = []
        for event in events:
          wave_starts: dict[tuple[int, int, int], int] = {}
          for p in decode(event.blob):
            if isinstance(p, (WAVESTART, WAVESTART_L4)): wave_starts[(p.wave, p.simd, p.cu)] = p._time
            elif isinstance(p, WAVEEND) and (key := (p.wave, p.simd, p.cu)) in wave_starts:
              our_waves.append((wave_starts[key], p._time))
        self.assertEqual(sorted(our_waves), sorted(roc_waves), f"wave times mismatch in {name}")

  def test_rocprof_inst_times_match(self):
    """Instruction times must match rocprof exactly (excluding s_endpgm)."""
    for name, (events, lib, base) in self.examples.items():
      with self.subTest(example=name):
        _, wave_insts = run_rocprof_decoder([e.blob for e in events], lib, base, self.target)
        # skip last inst per wave (s_endpgm) - it needs special handling (time + duration instead of time + stall)
        roc_insts = [time + stall for insts in wave_insts for time, stall in insts[:-1]]
        # extract from our decoder
        our_insts: list[int] = []
        for event in events:
          for p in decode(event.blob):
            if isinstance(p, INST) and p.op not in OTHER_SIMD_OPS: our_insts.append(p._time)
            elif isinstance(p, INST_L4) and p.op not in OTHER_SIMD_OPS_L4: our_insts.append(p._time)
            elif isinstance(p, VALUINST): our_insts.append(p._time)
            elif isinstance(p, IMMEDIATE): our_insts.append(p._time)
            elif isinstance(p, IMMEDIATE_MASK):
              for _ in range(bin(p.mask).count('1')): our_insts.append(p._time)
        self.assertEqual(sorted(our_insts), sorted(roc_insts), f"instruction times mismatch in {name}")

class TestSQTTExamplesRDNA3(SQTTExamplesTestBase):
  target = "gfx1100"
  expected = {
    "profile_empty_run_0": [1844, 1885, 1905, 1956, 1983, 1889],
    "profile_empty_run_1": [1780, 1885, 1905, 1956, 1983, 1889],
    "profile_gemm_run_0": [2656, 2025, 2045, 2096, 2123, 2029, 3183, 2019, 2039, 2090, 2117, 2023, 19119, 2013, 2033, 2084, 2111, 2017],
    "profile_gemm_run_1": [2662, 2025, 2045, 2096, 2123, 2029, 3179, 2019, 2039, 2090, 2117, 2023, 19113, 2071, 2091, 2142, 2169, 2075],
    "profile_plus_run_0": [1886, 2013, 2033, 2084, 2111, 2017],
    "profile_plus_run_1": [1988, 2071, 2091, 2142, 2169, 2075],
  }

class TestSQTTExamplesRDNA4(SQTTExamplesTestBase): target = "gfx1200"
# CDNA/MI300 (gfx950) uses a completely different 16-bit header packet format, not the nibble-based format.
# See decode_tt_header_stream in ghidra/librocprof-trace-decoder.c - it reads 16-bit headers and uses
# pkt_fmt = header & 0xf to look up packet_class (0x10=2bytes, 0x20=4bytes, 0x30=6bytes, 0x40=8bytes).
# This is NOT implemented yet - the nibble decoder produces garbage for CDNA data.
@unittest.skip("CDNA/MI300 uses 16-bit header format, not nibble-based - decoder not implemented")
class TestSQTTExamplesCDNA(SQTTExamplesTestBase):
  target = "gfx950"

if __name__ == "__main__":
  unittest.main()
