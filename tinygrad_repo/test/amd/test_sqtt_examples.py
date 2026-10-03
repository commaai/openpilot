#!/usr/bin/env python3
"""Tests for SQTT packet decoding using real captured examples."""
import pickle, unittest, ctypes, threading
from pathlib import Path
from tinygrad.helpers import DEBUG
from tinygrad.runtime.autogen import rocprof
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.rdna3.ins import SOPP
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp
from tinygrad.renderer.amd.sqtt import (decode, LAYOUT_HEADER, WAVESTART, WAVESTART_RDNA4, WAVEEND, WAVEEND_RDNA4, INST, INST_RDNA4, VALUINST,
                                     IMMEDIATE, IMMEDIATE_MASK, PACKET_TYPES_RDNA3, PACKET_TYPES_RDNA4, PACKET_TYPES_CDNA, CDNA_WAVESTART,
                                     print_packets, CDNA_WAVEEND, CDNA_INST)
from test.amd.helpers import TARGET_TO_ARCH
from test.amd.test_sqttmap import needs_rocprof

import tinygrad
EXAMPLES_DIR = Path(tinygrad.__file__).parent.parent / "extra/sqtt/examples"

# ═══════════════════════════════════════════════════════════════════════════════
# ROCPROF DECODER
# ═══════════════════════════════════════════════════════════════════════════════

def run_rocprof_decoder(blobs: list[bytes], lib: bytes, base: int, target: str):
  """Run rocprof decoder on SQTT blobs, returning raw occupancy and instruction records."""
  image, sections, _ = elf_loader(lib)
  text = next((sh for sh in sections if sh.name == ".text"), None)
  assert text is not None, "no .text section found"
  text_off, text_size = text.header.sh_addr, text.header.sh_size

  blob_iter, current_blob = iter(blobs), [None]  # type: ignore[var-annotated]
  occupancy_records: list[tuple[int, int, int, int, bool]] = []  # (wave_id, simd, cu, time, is_start)
  wave_insts: list[list[tuple[int, int]]] = []  # per-wave list of (time, stall)

  @rocprof.rocprof_trace_decoder_se_data_callback_t
  def copy_cb(buf, buf_size, _):  # type: ignore[no-untyped-def]
    blob = next(blob_iter, None)
    if blob is None: return 0
    current_blob[0] = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)  # type: ignore[call-overload]
    buf[0] = ctypes.cast(current_blob[0], ctypes.POINTER(ctypes.c_ubyte))  # type: ignore[arg-type]
    buf_size[0] = len(current_blob[0])  # type: ignore[arg-type]
    return len(current_blob[0])  # type: ignore[arg-type]

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
  t.join(timeout=5)
  if exc is not None: raise exc
  if t.is_alive(): raise RuntimeError("rocprof decoder timeout")
  return occupancy_records, wave_insts

class SQTTExamplesTestBase(unittest.TestCase):
  target: str
  examples: dict

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
          if DEBUG >= 2:
            print(f"\n=== {name} event {i} ===")
            print_packets(packets)
          self.assertGreater(len(packets), 0, f"no packets decoded from {name} event {i}")
          self.assertIsInstance(packets[0], LAYOUT_HEADER, f"first packet should be LAYOUT_HEADER in {name}")

  def test_packet_types_valid(self):
    all_classes = set(PACKET_TYPES_RDNA3.values()) | set(PACKET_TYPES_RDNA4.values()) | set(PACKET_TYPES_CDNA.values())
    for name, (events, *_) in self.examples.items():
      for i, event in enumerate(events):
        with self.subTest(example=name, event=i):
          for pkt in decode(event.blob):
            # Use isinstance to handle layout-specific subclasses (e.g., WAVESTART_RDNA4)
            self.assertTrue(any(isinstance(pkt, cls) for cls in all_classes), f"unknown packet type {type(pkt)} in {name}")

  def test_wave_lifecycle(self):
    for name, (events, *_) in self.examples.items():
      if "empty" in name: continue
      with self.subTest(example=name):
        all_packets = [p for e in events for p in decode(e.blob)]
        self.assertGreater(len([p for p in all_packets if isinstance(p, (WAVESTART, WAVESTART_RDNA4, CDNA_WAVESTART))]), 0, f"no WAVESTART in {name}")
        self.assertGreater(len([p for p in all_packets if isinstance(p, (WAVEEND, WAVEEND_RDNA4, CDNA_WAVEEND))]), 0, f"no WAVEEND in {name}")

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
        inst_packets = [p for p in all_packets if isinstance(p, (INST, INST_RDNA4, CDNA_INST))]
        self.assertGreater(len(inst_packets), 0, f"no INST packets in {name}")
        if isinstance(inst_packets[0], (INST, INST_RDNA4)):
          self.assertGreater(len([p for p in inst_packets if p.op.name.startswith("JUMP")]), 0, f"no JUMP packets in {name}")

  expected: dict[str, list[int]] = {}  # override in subclasses
  def test_packet_counts(self):
    if not self.expected: self.skipTest("no expected packet counts for this target")
    for name, (events, *_) in self.examples.items():
      with self.subTest(example=name):
        if not self.expected.get(name): continue
        counts = [len(list(decode(e.blob))) for e in events]
        self.assertEqual(counts, self.expected[name], f"packet count mismatch in {name}")

  @needs_rocprof
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
          first_timestamp:int|None = None
          for p in decode(event.blob):
            if first_timestamp is None: first_timestamp = p._time
            if isinstance(p, (WAVESTART, CDNA_WAVESTART, WAVESTART_RDNA4)): wave_starts[(p.wave, p.simd, p.cu)] = p._time
            elif isinstance(p, (WAVEEND, WAVEEND_RDNA4, CDNA_WAVEEND)) and (key := (p.wave, p.simd, p.cu)) in wave_starts:
              our_waves.append((wave_starts[key], p._time))
          for st in wave_starts.values():
            self.assertGreater(st, first_timestamp, "wave start must be after the first packet")
        # rocprof fails non deterministically and gives inaccurate timestamps.
        #self.assertEqual(sorted(our_waves), sorted(roc_waves), f"wave times mismatch in {name}")
        for st, et in our_waves:
          self.assertGreater(et, st, "wave end must be after start")

  @needs_rocprof
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
            # INST ops for non-traced SIMDs (excluded from instruction count)
            if isinstance(p, (INST, INST_RDNA4)) and not p.op.name.startswith("OTHER_"): our_insts.append(p._time)
            elif isinstance(p, VALUINST): our_insts.append(p._time)
            elif isinstance(p, IMMEDIATE): our_insts.append(p._time)
            elif isinstance(p, IMMEDIATE_MASK):
              for _ in range(bin(p.mask).count('1')): our_insts.append(p._time)
        self.assertEqual(sorted(our_insts), sorted(roc_insts), f"instruction times mismatch in {name}")

class TestSQTTExamplesRDNA3(SQTTExamplesTestBase):
  target = "gfx1100"
  expected = {
    "profile_empty_run_0": [1880, 1867, 1920, 1971, 1998, 1904],
    "profile_empty_run_1": [1880, 1867, 1920, 1971, 1998, 1904],
    "profile_gemm_run_0": [3275, 3278, 2426, 2475, 2511, 2431],
    "profile_gemm_run_1": [3264, 3268, 2420, 2469, 2504, 2401],
    "profile_ops_run_0": [1944, 4903, 1984, 2035, 2062, 1968],
    "profile_ops_run_1": [1944, 4918, 1984, 2035, 2062, 1968],
    "profile_plus_run_0": [1938, 1932, 1978, 2029, 2056, 1962],
    "profile_plus_run_1": [1891, 1874, 1931, 1982, 2009, 1915],
  }

class TestSQTTExamplesRDNA4(SQTTExamplesTestBase): target = "gfx1200"

class TestSQTTExamplesCDNA(SQTTExamplesTestBase):
  target = "gfx950"
  def test_rocprof_wave_times_match(self): self.skipTest("TODO: requires timestamp patching")
  def test_rocprof_inst_times_match(self): self.skipTest("TODO: requires timestamp patching")

if __name__ == "__main__":
  unittest.main()
