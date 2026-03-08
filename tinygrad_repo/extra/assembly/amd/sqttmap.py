# maps SQTT trace packets to instructions.
from dataclasses import dataclass
from typing import Iterator

from extra.assembly.amd.sqtt import decode, print_packets, INST, VALUINST, IMMEDIATE, WAVESTART, WAVEEND, InstOp, PacketType, IMMEDIATE_MASK
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.autogen.rdna3.ins import SOPP, s_endpgm
from extra.assembly.amd.autogen.rdna3.enum import SOPPOp

@dataclass(frozen=True)
class InstructionInfo:
  pc: int
  wave: int
  inst: Inst

def map_insts(data:bytes, lib:bytes, target:int) -> Iterator[tuple[PacketType, InstructionInfo|None]]:
  """maps SQTT packets to instructions, yields (packet, instruction_info or None)"""
  # map pcs to insts
  from tinygrad.viz.serve import amd_decode
  pc_map = amd_decode(lib, target)

  wave_pc:dict[int, int] = {}
  # only processing packets on one [CU, SIMD] unit
  def simd_select(p) -> bool: return getattr(p, "cu", 0) == 0 and getattr(p, "simd", 0) == 0
  for p in decode(data):
    if not simd_select(p): continue
    if isinstance(p, WAVESTART):
      assert p.wave not in wave_pc, "only one inflight wave per unit"
      wave_pc[p.wave] = next(iter(pc_map))
      continue
    if isinstance(p, WAVEEND):
      pc = wave_pc.pop(p.wave)
      yield (p, InstructionInfo(pc, p.wave, s_endpgm()))
      continue
    # skip OTHER_ instructions, they don't belong to this unit
    if isinstance(p, INST) and p.op.name.startswith("OTHER_"): continue
    if isinstance(p, IMMEDIATE_MASK):
      # immediate mask may yield multiple times per packet
      for wave in range(16):
        if p.mask & (1 << wave):
          inst = pc_map[pc:=wave_pc[wave]]
          # can this assert be more strict?
          assert isinstance(inst, SOPP), f"IMMEDIATE_MASK packet must map to SOPP, got {inst}"
          wave_pc[wave] += inst.size()
          yield (p, InstructionInfo(pc, wave, inst))
      continue
    if isinstance(p, (VALUINST, INST, IMMEDIATE)):
      inst = pc_map[pc:=wave_pc[p.wave]]
      # s_delay_alu doesn't get a packet?
      if isinstance(inst, SOPP) and inst.op in {SOPPOp.S_DELAY_ALU}:
        wave_pc[p.wave] += inst.size()
        inst = pc_map[pc:=wave_pc[p.wave]]
      # identify a branch instruction, only used for asserts
      is_branch = isinstance(inst, SOPP) and "BRANCH" in inst.op_name
      if is_branch: assert isinstance(p, INST) and p.op in {InstOp.JUMP_NO, InstOp.JUMP}, f"branch can only be folowed by jump packets, got {p}"
      # JUMP handling
      if isinstance(p, INST) and p.op is InstOp.JUMP:
        assert is_branch, f"JUMP packet must map to a branch instruction, got {inst}"
        x = inst.simm16 & 0xffff
        wave_pc[p.wave] += inst.size() + (x - 0x10000 if x & 0x8000 else x)*4
      else:
        if is_branch: assert inst.op != SOPPOp.S_BRANCH, f"S_BRANCH must have a JUMP packet, got {p}"
        wave_pc[p.wave] += inst.size()
      yield (p, InstructionInfo(pc, p.wave, inst))
      continue
    # for all other packets (VMEMEXEC, ALUEXEC, etc.), yield with None
    yield (p, None)

# test to compare every packet with the rocprof decoder

def test_rocprof_inst_traces_match(sqtt, prg, target):
  from tinygrad.viz.serve import amd_decode
  from extra.sqtt.roc import decode as roc_decode, InstExec
  addr_table = amd_decode(prg.lib, target)
  disasm = {addr+prg.base:(inst.disasm(), inst.size()) for addr,inst in addr_table.items()}
  rctx = roc_decode([sqtt], {prg.tag:disasm})
  rwaves = rctx.inst_execs.get((sqtt.kern, sqtt.exec_tag), [])
  rwaves_iter:dict[int, list[Iterator[InstExec]]] = {} # wave unit (0-15) -> list of inst trace iterators for all executions on that unit
  for w in rwaves: rwaves_iter.setdefault(w.wave_id, []).append(w.unpack_insts())

  passed_insts = 0
  for pkt, info in map_insts(sqtt.blob, prg.lib, target):
    if DEBUG >= 2: print_packets([pkt])
    if info is None: continue
    if DEBUG >= 2: print(f"{' '*29}{info.inst.disasm()}")
    rocprof_inst = next(rwaves_iter[info.wave][0])
    ref_pc = rocprof_inst.pc-prg.base
    # always check pc matches
    assert ref_pc == info.pc, f"pc mismatch {ref_pc}:{disasm[rocprof_inst.pc][0]} != {info.pc}:{info.inst.disasm()}"
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

  if len(rwaves):
    print(f"passed for {passed_insts} instructions across {len(rwaves)} waves scheduled on {len(rwaves_iter)} wave units")

if __name__ == "__main__":
  import argparse, pickle, pathlib
  from tinygrad.helpers import temp, DEBUG
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', type=pathlib.Path, metavar="PATH", help='Path to profile (optional file, default: latest profile)',
                      default=pathlib.Path(temp("profile.pkl", append_user=True)))
  parser.add_argument('--kernel', type=str, default=None, metavar="NAME", help='Kernel to focus on (optional name, default: all kernels)')
  args = parser.parse_args()
  with open(args.profile, "rb") as f:
    data = pickle.load(f)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  kern_events = {e.tag:e for e in data if type(e).__name__ == "ProfileProgramEvent"}
  target = next((e for e in data if type(e).__name__ == "ProfileDeviceEvent" and e.device.startswith("AMD"))).props["gfx_target_version"]
  for e in sqtt_events:
    if args.kernel is not None and args.kernel != e.kern: continue
    if not e.itrace: continue
    print(f"==== {e.kern}")
    test_rocprof_inst_traces_match(e, kern_events[e.kern], target)
