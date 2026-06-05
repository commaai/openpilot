# test to compare every packet with the rocprof decoder
import unittest, pickle, functools, json
from typing import Iterator
from pathlib import Path
from tinygrad.helpers import DEBUG, getenv, temp, ansistrip, Context
from tinygrad.renderer.amd.sqtt import print_packets, map_insts
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
from tinygrad.viz.serve import sqtt_timeline, amd_decode
from test.amd.disasm import disasm
from test.null.test_viz import run_cli

import tinygrad
EXAMPLES_DIR = Path(tinygrad.__file__).parent.parent / "extra/sqtt/examples"

def needs_rocprof(fn):
  @functools.wraps(fn)
  def wrapper(self, *args, **kwargs):
    # check if latest rocprof is available, if not, skip rocprof comparison tests
    # rocprof doesn't have a version string, decode a known pickle to validate it's the latest
    try:
      from extra.sqtt.roc import decode as roc_decode
      with open(EXAMPLES_DIR/"gfx1200"/"profile_plus_run_0.pkl", "rb") as f:
        data = pickle.load(f)
      sqtt = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"][1]
      kern = {e.tag:e for e in data if type(e).__name__ == "ProfileProgramEvent"}[sqtt.kern]
      rctx = roc_decode([sqtt], {kern.tag:{addr+kern.base:inst for addr,inst in amd_decode(kern.lib, "gfx1200").items()}})
      insts = [e.time for e in list(rctx.inst_execs.values())[0][0].unpack_insts()]
      self.assertListEqual(insts, [28178, 28179, 28180, 28181, 28182, 29882, 29883, 29884, 29885, 30966, 30983, 30985, 30992, 30993])
    except Exception as e: self.skipTest(f"latest rocprof not available, install with extra/sqtt/install_rocprof_decoder.py: {e}")
    return fn(self, *args, **kwargs)
  return wrapper

def rocprof_inst_traces_match(sqtt, prg, target):
  from extra.sqtt.roc import decode as roc_decode, InstExec
  addr_table = amd_decode(prg.lib, target)
  disasm_map = {addr+prg.base:inst for addr,inst in addr_table.items()}
  rctx = roc_decode([sqtt], {prg.tag:disasm_map})
  rwaves = rctx.inst_execs.get((sqtt.kern, sqtt.exec_tag), [])
  rwaves_iter:dict[int, list[Iterator[InstExec]]] = {} # wave unit (0-15) -> list of inst trace iterators for all executions on that unit
  for w in rwaves: rwaves_iter.setdefault(w.wave_id, []).append(w.unpack_insts())

  if not rwaves: return 0, 0, 0

  passed_insts = 0
  for pkt, info in map_insts(sqtt.blob, prg.lib, target):
    if DEBUG >= 2: print_packets([(pkt, info)])
    if info is None: continue
    if DEBUG >= 2: print(f"{' '*29}{disasm(info.inst)}")
    rocprof_inst = next(rwaves_iter[info.wave][0])
    ref_pc = rocprof_inst.pc-prg.base
    # always check pc matches
    assert ref_pc == info.pc, f"pc mismatch {ref_pc}:{disasm_map[rocprof_inst.pc]} != {info.pc}:{disasm(info.inst)}"
    # special handling for s_endpgm, it marks the wave completion.
    if info.inst == s_endpgm():
      completed_wave = list(rwaves_iter[info.wave].pop(0))
      assert len(completed_wave) == 0, f"incomplete instructions in wave {info.wave}"
    # otherwise the packet timestamp is time + "stall"
    else:
      assert pkt._time == rocprof_inst.time+rocprof_inst.stall
    passed_insts += 1

  for k,v in rwaves_iter.items():
    assert len(v) == 0, f"incomplete wave {k}"

  return passed_insts, len(rwaves), len(rwaves_iter)

class TestSQTTMapBase(unittest.TestCase):
  target: str
  examples: dict

  @classmethod
  def setUpClass(cls):
    if cls is TestSQTTMapBase: raise unittest.SkipTest("base class")
    cls.examples = {}
    for pkl_path in ([Path(temp("profile.pkl", append_user=True))] if getenv("LOAD_PROFILE") else sorted((EXAMPLES_DIR/cls.target).glob("*.pkl"))):
      with open(pkl_path, "rb") as f:
        data = pickle.load(f)
      sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
      kern_events = {e.tag:e for e in data if type(e).__name__ == "ProfileProgramEvent"}
      if sqtt_events and kern_events:
        cls.examples[pkl_path.stem] = (sqtt_events, kern_events, cls.target)

  @needs_rocprof
  def test_rocprof_inst_traces_match(self):
    for name, (events, kern_events, target) in self.examples.items():
      if "sync" in name and self.target.startswith("gfx12"):
        self.skipTest("our timestamps are off by a few cycles because rocprof patches timestamps for rdna4 barriers")
      for event in events:
        if not event.itrace: continue
        if event.kern not in kern_events: continue
        with self.subTest(example=name, kern=event.kern):
          passed_insts, n_waves, n_units = rocprof_inst_traces_match(event, kern_events[event.kern], target)
          if n_waves: print(f"{name}: passed for {passed_insts} instructions across {n_waves} waves scheduled on {n_units} wave units")

  def test_sqtt_timeline(self):
    for name, (events, kern_events, target) in self.examples.items():
      for event in events:
        if (p:=kern_events.get(event.kern)) is None: continue
        with self.subTest(example=name, kern=event.kern):
          # skip if there's no SQTT frequency data
          if not (timeline:=list(sqtt_timeline(event.blob, p.lib, target))): continue
          if not (frequency:=[e.key for e in timeline if type(e).__name__ == "ProfilePointEvent" and e.name == "freq_hz"]): continue
          mean = sum(frequency) / len(frequency)
          variance = sum((v - mean) ** 2 for v in frequency) / len(frequency)
          self.assertGreater(mean, 0)
          if DEBUG >= 2: print(f"{name:20s} SE:{event.se} {mean/1e9:.2f} GHz mean, {variance/1e18:.2f} GHz^2 variance")
          events = [e for e in timeline if type(e).__name__ == "ProfileRangeEvent"]
          insts, execs = 0, 0
          for e in events:
            if "EXEC" in e.device:
              if "ALT" not in e.name.display_name: execs += 1
            elif "WAVE" in e.device:
              # sopk/immediates don't get ALU/MEM EXEC
              if e.name.display_name not in {"IMMEDIATE", "IMMEDIATE_MASK", "JUMP", "JUMP_NO", "MESSAGE", "BARRIER", "BARRIER_SIGNAL",
                                             "WAVEEND", "WAVEEND_RDNA4", "WAVERDY"} and not e.name.display_name.startswith("OTHER_"): insts += 1
            else: raise Exception(f"timeline row must be INST or EXEC, got {e.device}")
          self.assertEqual(execs, insts)

  def test_wave_sync(self):
    for name, (events, kern_events, target) in self.examples.items():
      for event in events:
        wave_barriers = {}
        for e in sqtt_timeline(event.blob, kern_events[event.kern].lib, target):
          if type(e).__name__ == "ProfileRangeEvent" and e.name.display_name == "BARRIER": wave_barriers.setdefault(e.device, []).append(e)
        if not wave_barriers: continue
        for row, events in wave_barriers.items():
          for e in events:
            assert e.en-e.st > 1, f"all barriers must have a duration greater than 1, got {e}"

  def test_sqtt_cli(self):
    for pkl_path in sorted((EXAMPLES_DIR/self.target).glob("*.pkl")):
      out = run_cli("--profile-path", str(pkl_path), "--ls")
      sqtt_traces = [l["value"].strip() for l in out if "SQTT" in l["value"]]
      for name in sqtt_traces:
        lines = run_cli("--profile-path", str(pkl_path), "-s", ansistrip(name))
        self.assertIn("Clk", lines[0]["value"])
        waves = [r["clk"] for r in lines[2:] if "WAVE" in r["unit"]]
        self.assertEqual(waves, sorted(waves), f"wave timestamps not monotonic in {name}")
      with Context(DEBUG=2):
        kernels = run_cli("--profile-path", str(pkl_path), "-s", "AMD")
      self.assertEqual(len(kernels), len(self.examples[pkl_path.stem][1]))

class TestSQTTMapRDNA3(TestSQTTMapBase): target = "gfx1100"

class TestSQTTMapRDNA4(TestSQTTMapBase):
  target = "gfx1200"

  @unittest.expectedFailure
  def test_pipes(self):
    events, kernels, target = self.examples["profile_handwritten_run_0"]
    lib = list(kernels.values())[0].lib
    dispatch_st:dict[str, int] = {}
    row_ends:dict[str, int] = {}
    row_counts:dict[str, int] = {}
    for e in sqtt_timeline(events[1].blob, lib, target):
      if type(e).__name__ != "ProfileRangeEvent": continue
      info = json.loads(e.name.ret) if e.name.ret else {}
      if e.device.startswith("WAVE"):
        idx = row_counts.get(e.device, 0)
        dispatch_st[f"{e.device}-{idx}"] = int(e.st)
        row_counts[e.device] = idx + 1
      elif info.startswith("LINK:"):
        delay = int(e.st) - dispatch_st[info[len("LINK:"):]]
        self.assertGreaterEqual(delay, 1, f"EXEC {e.device} starts before DISPATCH: delay={delay}")
        if (prev_en:=row_ends.get(e.device)) is not None:
          self.assertGreaterEqual(e.st, prev_en, f"EXEC overlap in {e.device}: {e.st} < prev end {prev_en}")
        row_ends[e.device] = int(e.en)

class TestSQTTMapCDNA(TestSQTTMapBase):
  target = "gfx950"
  def test_rocprof_inst_traces_match(self): self.skipTest("requires timestamp patching to match rocprof, currently it's off by a few cycles")

if __name__ == "__main__":
  unittest.main()
