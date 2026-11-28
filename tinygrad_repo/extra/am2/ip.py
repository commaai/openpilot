from __future__ import annotations
import ctypes, dataclasses, functools, operator
from typing import Literal, Iterable
from tinygrad.uop.ops import UOp, Ops, uopfunc
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.helpers import to_mv, data64, lo32, hi32, DEBUG, pad_bytes, getbits, round_up, getenv
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.amd import import_soc
from tinygrad.runtime.support.hcq2 import get_time_ms, ccall, patch

# **** access

def u32(x:UOp|int) -> UOp: return x.cast(dtypes.uint32) if isinstance(x, UOp) else UOp.const(x, dtypes.uint32)
def u64(x:UOp|int) -> UOp: return x.cast(dtypes.uint64) if isinstance(x, UOp) else UOp.const(x, dtypes.uint64)
def i64(x:UOp|int) -> UOp: return x.cast(dtypes.int64) if isinstance(x, UOp) else UOp.const(x, dtypes.int64) # a bar is larger than 4GB

@uopfunc
def am_poke(mem:UOp, i:UOp, val:UOp, bus:UOp) -> UOp: # a volatile write. nothing is written once the program failed
  mem = mem.replace(arg=dataclasses.replace(mem.arg, volatile=True))
  live = UOp.range(bus.index(5).load().eq(0).cast(dtypes.int), next(UOp.unique_num), dtype=dtypes.int)
  return mem.after(live).index(i).store(val).end(live).sink()

@uopfunc
def am_peek(mem:UOp, i:UOp, out:UOp) -> UOp: # a volatile read
  mem = mem.replace(arg=dataclasses.replace(mem.arg, volatile=True))
  return out.index(0).store(mem.index(i).load()).sink()

def am_store(bus:UOp, mem:UOp, off:UOp|int, val:UOp) -> UOp: # the bus after a write to device memory at a byte offset
  return bus.after(am_poke(mem.after(bus).bitcast(val.dtype), i64(off) // val.dtype.itemsize, val, bus))

def am_load(bus:UOp, mem:UOp, off:UOp|int, dt:DType=dtypes.uint32) -> tuple[UOp, UOp]: # the bus after a read, and what was read
  slot = UOp.placeholder((1,), dt, addrspace=AddrSpace.REG)
  bus = bus.after(am_peek(mem.after(bus).bitcast(dt), i64(off) // dt.itemsize, slot.index(0)))
  return bus, slot.after(bus).index(0).load()

def am_memcpy(bus:UOp, adev, mem:UOp, off:UOp|int, data:bytes, rows=()) -> UOp: # rows: words the program fills in
  n = len(data:=data.ljust(round_up(len(data), 4), b'\0'))
  src = patch(UOp.alloc((n,), dtypes.uint8, device=adev.cpu, name="am_data"), rows, data).index(0)
  return bus.after(ccall(libc.memcpy, mem.after(bus).index(i64(off)), src, u64(n)))

def am_out(bus:UOp, i:int, val:UOp) -> UOp: return bus.after(bus.index(8 + i).store(val.cast(dtypes.uint64))) # a number for python

def am_rreg(bus:UOp, adev, reg:int, inst=0, direct=False) -> tuple[UOp, UOp]: # AMDev.rreg
  if reg >= len(adev.mmio): return RREG32_PCIE(bus, adev, reg)
  return am_load(bus, adev.rmmio, reg * 4)

def am_wreg(bus:UOp, adev, reg:UOp|int, val:UOp|int, inst=0, direct=False) -> UOp: # AMDev.wreg. a register read from another is a UOp
  if isinstance(reg, int) and reg >= len(adev.mmio): return WREG32_PCIE(bus, adev, reg, val)
  return am_store(bus, adev.rmmio, i64(reg) * 4, u32(val))

def RREG32(bus:UOp, reg, inst=0, direct=False) -> tuple[UOp, UOp]: return am_rreg(bus, reg.adev, reg.addr[inst], inst, direct) # reg.read

def WREG32(bus:UOp, reg, _am_val:UOp|int=0, inst=0, direct=False, **kwargs) -> UOp: # reg.write. a field can be a value of the program
  val = functools.reduce(operator.or_, (v << reg.fields[k][0] for k, v in kwargs.items()), _am_val)
  return am_wreg(bus, reg.adev, reg.addr[inst], val, inst, direct)

def WREG32_FIELD(bus:UOp, reg, inst=0, **kwargs) -> UOp: # reg.update
  bus, val = RREG32(bus, reg, inst)
  return WREG32(bus, reg, val & (~reg.fields_mask(*kwargs.keys()) & 0xffffffff), inst, **kwargs)

def WREG32_PAIR(bus:UOp, adev, reg_base:str, lo_suffix:str, hi_suffix:str, val:int, inst=0, direct=False) -> UOp: # AMDev.wreg_pair
  bus = WREG32(bus, adev.reg(f"{reg_base}{lo_suffix}"), lo32(val), inst, direct)
  return WREG32(bus, adev.reg(f"{reg_base}{hi_suffix}"), hi32(val), inst, direct)

def RREG32_PCIE(bus:UOp, adev, reg:int) -> tuple[UOp, UOp]:
  bus = WREG32(bus, adev.reg("regBIF_BX_PF0_RSMU_INDEX"), reg * 4)
  return RREG32(bus, adev.reg("regBIF_BX_PF0_RSMU_DATA"))

def WREG32_PCIE(bus:UOp, adev, reg:int, val:UOp|int) -> UOp:
  bus = WREG32(bus, adev.reg("regBIF_BX_PF0_RSMU_INDEX"), reg * 4)
  return WREG32(bus, adev.reg("regBIF_BX_PF0_RSMU_DATA"), val)

def WREG32_PCIE_EXT(bus:UOp, adev, reg:int, val:int, aid:int=0) -> UOp:
  reg_addr = reg * 4 + ((((aid & 0b11) << 32) | (1 << 34)) if aid > 0 else 0)
  bus = WREG32(bus, adev.reg("regBIF_BX0_PCIE_INDEX2"), lo32(reg_addr))
  if hi32(reg_addr) > 0: bus = WREG32(bus, adev.reg("regBIF_BX0_PCIE_INDEX2_HI"), hi32(reg_addr) & 0xff)
  bus = WREG32(bus, adev.reg("regBIF_BX0_PCIE_DATA2"), val)
  return WREG32(bus, adev.reg("regBIF_BX0_PCIE_INDEX2_HI"), 0) if hi32(reg_addr) > 0 else bus

def sleep(bus:UOp, seconds:float) -> UOp: return bus.after(ccall(libc.usleep, UOp.const(int(seconds * 1e6), dtypes.uint32).after(bus)))

def am_fail(bus:UOp, adev, cond:UOp, error:Exception) -> UOp: # if cond the program failed: python raises the first error of a program
  adev.errors.append(error)
  return bus.after(bus.index(UOp.const(5).valid(bus.index(5).load().eq(0))).store(cond.cast(dtypes.uint64) * len(adev.errors)))

def wait_cond(bus:UOp, adev, cb, mask:int|None=None, value:UOp|int|bool=True, timeout_ms:int=10000, msg:str|None="") -> UOp: # None: may time out
  start, miss = get_time_ms(bus), UOp.placeholder((1,), dtypes.bool, addrspace=AddrSpace.REG)
  loop = UOp.range(UOp(Ops.NOOP).after(bus, start), next(UOp.unique_num), dtype=dtypes.void)
  polled, val = cb(bus.after(loop))
  polled = miss.after(polled).index(0).store((val if mask is None else val & mask) != value)
  alive = bus.after(polled).index(5).load().eq(0) & (get_time_ms(polled) - start < timeout_ms)
  bus = bus.after(polled.backedge(loop, miss.after(polled).index(0).load() & alive))
  if msg is None: return bus
  return am_fail(bus, adev, miss.after(bus).index(0).load(), TimeoutError(f"{msg}. Timed out after {timeout_ms} ms"))

# **** ip blocks

class AM_IP:
  def __init__(self, adev): self.adev = adev
  def init_sw(self): pass # Prepare sw/allocations for this IP

# **** soc

class AM_SOC(AM_IP):
  def init_sw(self):
    self.module = import_soc(self.adev.ip_ver[am.GC_HWIP])
    self.ih_clients = am.enum_soc21_ih_clientid if (ih_soc21:=self.adev.ip_ver[am.GC_HWIP][0] >= 11) else am.enum_soc15_ih_clientid

    self.gfx_ih_clients = [am.SOC21_IH_CLIENTID_GRBM_CP, am.SOC21_IH_CLIENTID_GFX] \
      if ih_soc21 else [am.SOC15_IH_CLIENTID_GRBM_CP] + [getattr(am, f'SOC15_IH_CLIENTID_SE{i}SH') for i in range(4)]
    self.sdma_ih_clients = [] if ih_soc21 else [getattr(am, f'SOC15_IH_CLIENTID_SDMA{i}') for i in range(8)]

    def _ih_srcs(pref:str, hwip:int) -> dict[int, str]:
      return {getattr(am, k): k[off+9:] for k in dir(am) if k.startswith(f'{pref}_{self.adev.ip_ver[hwip][0]}') and (off:=k.find('__SRCID__')) != -1}

    gfx_srcs, sdma_srcs = _ih_srcs('GFX', am.GC_HWIP), _ih_srcs('SDMA0', am.SDMA0_HWIP)
    self.ih_srcs_names:dict[int, dict[int, str]] = {**{k: gfx_srcs for k in self.gfx_ih_clients}, **{k: sdma_srcs for k in self.sdma_ih_clients}}

@uopfunc
def soc_init_hw(bus:UOp, adev) -> UOp:
  if adev.ip_ver[am.NBIO_HWIP] in {(7,9,0), (7,9,1)}:
    # fence doorbells for harvested xccs (0xff & ~xcc_mask in the kernel); a fully-unharvested chip keeps the previous 0x0
    live_xccs = sum(1 << i for i in adev.regs_offset[am.GC_HWIP] if i not in adev.harvested[am.GC_HWIP] and i < 8)
    bus = WREG32(bus, adev.regXCC_DOORBELL_FENCE, 0xff & ~live_xccs)
    for aid in adev.aids[1:]:
      bus = WREG32_PCIE_EXT(bus, adev, adev.regXCC_DOORBELL_FENCE.addr[0], adev.regXCC_DOORBELL_FENCE.encode(shub_slv_mode=1), aid=aid)
    bus = WREG32(bus, adev.regBIFC_GFX_INT_MONITOR_MASK, 0x7ff)
    bus = WREG32(bus, adev.regBIFC_DOORBELL_ACCESS_EN_PF, 0xfffff)
  else: bus = WREG32_FIELD(bus, adev.regRCC_DEV0_EPF2_STRAP2, strap_no_soft_reset_dev0_f2=0x0)
  bus = WREG32(bus, adev.regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN, 0x1)
  return bus.sink()

@uopfunc
def soc_set_clockgating_state(bus:UOp, adev) -> UOp:
  if adev.ip_ver[am.HDP_HWIP] >= (5,2,1): bus = WREG32_FIELD(bus, adev.regHDP_MEM_POWER_CTRL, atomic_mem_power_ctrl_en=1, atomic_mem_power_ds_en=1)
  return bus.sink()

def soc_doorbell_enable(bus:UOp, adev, port, awid=0, awaddr_31_28_value=0, offset=0, size=0, aid=0) -> UOp:
  reg = adev.reg(f"{'regGDC_S2A0_S2A' if adev.ip_ver[am.GC_HWIP] >= (12,0,0) else 'regS2A'}_DOORBELL_ENTRY_{port}_CTRL")
  val = reg.encode(**{f"s2a_doorbell_port{port}_enable":1, f"s2a_doorbell_port{port}_awid":awid,  f"s2a_doorbell_port{port}_range_size":size,
    f"s2a_doorbell_port{port}_awaddr_31_28_value":awaddr_31_28_value, f"s2a_doorbell_port{port}_range_offset":offset})

  if adev.ip_ver[am.NBIO_HWIP] in {(7,9,0), (7,9,1)}: return WREG32_PCIE_EXT(bus, adev, reg.addr[0], val, aid=aid)
  else: return WREG32(bus, reg, val)

# **** gmc

class AM_GMC(AM_IP):
  def init_sw(self):
    self.vmhubs = len(self.adev.regs_offset[am.MMHUB_HWIP])

    xgmi_lfb_cntl = self.adev.regGCMC_VM_XGMI_LFB_CNTL.read_bitfields() if hasattr(self.adev, 'regGCMC_VM_XGMI_LFB_CNTL') else {}
    self.xgmi_phys_id, self.xgmi_max_region = xgmi_lfb_cntl.get('pf_lfb_region', 0), xgmi_lfb_cntl.get('pf_max_region', 0)
    self.xgmi_seg_sz = self.adev.regGCMC_VM_XGMI_LFB_SIZE.read_bitfields()['pf_lfb_size']<<24 if hasattr(self.adev, 'regGCMC_VM_XGMI_LFB_CNTL') else 0

    self.paddr_base = self.xgmi_phys_id * self.xgmi_seg_sz

    self.fb_base = (self.adev.regMMMC_VM_FB_LOCATION_BASE.read() & 0xFFFFFF) << 24
    self.fb_end = (self.adev.regMMMC_VM_FB_LOCATION_TOP.read() & 0xFFFFFF) << 24

    # Memory controller aperture
    self.mc_base = self.fb_base + self.paddr_base

    # VM aperture
    self.vm_base = self.adev.mm.va_base
    self.vm_end = min(self.vm_base + (1 << self.adev.mm.va_bits) - 1, 0x7fffffffffff)

    self.trans_futher = self.adev.ip_ver[am.GC_HWIP] < (10, 0, 0)

    # mi3xx has 48-bit, others have 44-bit address space
    self.address_space_mask = (1 << (48 if self.adev.ip_ver[am.GC_HWIP][:2] in {(9,4), (9,5)} else 44)) - 1

    self.memscratch_xgmi_paddr = self.adev.paddr2xgmi(self.adev.mm.palloc(0x1000, zero=False, boot=True))
    self.dummy_page_xgmi_paddr = self.adev.paddr2xgmi(self.adev.mm.palloc(0x1000, zero=False, boot=True))

    # MM hub is inited before any tlb flushes and is still valid during partial_boot, so set it to true
    self.hub_initted = {"MM": True, "GC": False}

    self.mm_insts = self.adev.aids if self.adev.ip_ver[am.NBIO_HWIP] in {(7,9,0), (7,9,1)} else list(range(self.vmhubs)) # dead mmhubs hang us

    self.pf_status_reg = lambda ip: f"reg{ip}VM_L2_PROTECTION_FAULT_STATUS{'_LO32' if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else ''}"

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_pte_flags(self, pte_lv, is_table, frag, uncached, system, snooped, valid, extra=0):
    extra |= (am.AMDGPU_PTE_SYSTEM * system) | (am.AMDGPU_PTE_SNOOPED * snooped) | (am.AMDGPU_PTE_VALID * valid) | am.AMDGPU_PTE_FRAG(frag)
    if not is_table: extra |= (am.AMDGPU_PTE_WRITEABLE | am.AMDGPU_PTE_READABLE | am.AMDGPU_PTE_EXECUTABLE)
    if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0):
      extra |= am.AMDGPU_PTE_MTYPE_GFX12(0, self.adev.soc.module.MTYPE_UC if uncached else 0)
      extra |= (am.AMDGPU_PDE_PTE_GFX12 if not is_table and pte_lv != am.AMDGPU_VM_PTB else (am.AMDGPU_PTE_IS_PTE if not is_table else 0))
    elif self.adev.ip_ver[am.GC_HWIP] >= (10,0,0):
      extra |= am.AMDGPU_PTE_MTYPE_NV10(0, self.adev.soc.module.MTYPE_UC if uncached else 0)
      extra |= (am.AMDGPU_PDE_PTE if not is_table and pte_lv != am.AMDGPU_VM_PTB else 0)
    else:
      extra |= am.AMDGPU_PTE_MTYPE_VG10(0, self.adev.soc.module.MTYPE_UC if uncached else 0)
      if is_table and pte_lv == am.AMDGPU_VM_PDB1: extra |= am.AMDGPU_PDE_BFS(0x9)
      if is_table and pte_lv == am.AMDGPU_VM_PDB0: extra |= am.AMDGPU_PTE_TF
      if not is_table and pte_lv not in {am.AMDGPU_VM_PTB, am.AMDGPU_VM_PDB0}: extra |= am.AMDGPU_PDE_PTE
    return extra
  def is_pte_huge_page(self, pte_lv, pte):
    if self.adev.ip_ver[am.GC_HWIP] < (10,0,0): return (pte & am.AMDGPU_PDE_PTE) if pte_lv != am.AMDGPU_VM_PDB0 else not (pte & am.AMDGPU_PTE_TF)
    return pte & (am.AMDGPU_PDE_PTE_GFX12 if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else am.AMDGPU_PDE_PTE)

  def flush_tlb(self, ip:Literal["MM", "GC"], vmid, flush_type=0): self.adev.run(gmc_flush_tlb, ip, vmid, flush_type, cache=self.hub_initted[ip])

@uopfunc
def gmc_init_hw(bus:UOp, adev) -> UOp: return bus.after(gmc_init_hub(bus, adev, "MM", insts=adev.gmc.mm_insts)).sink()

def gmc_flush_hdp(bus:UOp, adev) -> UOp:
  bus, reg = RREG32(bus, adev.reg("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"))
  return am_wreg(bus, adev, reg // 4, 0x0)

@uopfunc
def gmc_flush_tlb(bus:UOp, adev, ip:Literal["MM", "GC"], vmid, flush_type=0) -> UOp:
  bus = gmc_flush_hdp(bus, adev)

  # Can't issue TLB invalidation if the hub isn't initialized.
  if not adev.gmc.hub_initted[ip]: return bus.sink()

  req = adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_REQ").encode(flush_type=flush_type, per_vmid_invalidate_req=(1 << vmid),
    invalidate_l2_ptes=1, invalidate_l2_pde0=1, invalidate_l2_pde1=1, invalidate_l2_pde2=1, invalidate_l1_ptes=1,
    clear_protection_fault_status_addr=0)

  for inst in (adev.gmc.mm_insts if ip == "MM" else range(adev.gfx.xccs)):
    if ip == "MM":
      bus = wait_cond(bus, adev, lambda bus: RREG32(bus, adev.regMMVM_INVALIDATE_ENG17_SEM, inst=inst), mask=0x1, value=1, msg="mm flush_tlb timeout")

    bus = WREG32(bus, adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_REQ"), req, inst=inst)

    bus = wait_cond(bus, adev, lambda bus: RREG32(bus, adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_ACK"), inst=inst), mask=(1 << vmid), value=(1 << vmid),
                    msg="flush_tlb timeout")

    if ip == "MM": bus = WREG32(bus, adev.regMMVM_INVALIDATE_ENG17_SEM, 0x0, inst=inst)
    if adev.ip_ver[am.GC_HWIP] >= (11,0,0) and ip == "MM":
      bus = WREG32_FIELD(bus, adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2, reserved_cache_private_invalidation=1, inst=inst)

      # Read back the register to ensure the invalidation is complete
      bus, _ = RREG32(bus, adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2, inst=inst)
  return bus.sink()

def gmc_enable_vm_addressing(bus:UOp, adev, page_table, ip:Literal["MM", "GC"], vmid, inst) -> UOp:
  bus = WREG32_PAIR(bus, adev, f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_START_ADDR", "_LO32", "_HI32", adev.gmc.vm_base >> 12, inst=inst)
  bus = WREG32_PAIR(bus, adev, f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_END_ADDR", "_LO32", "_HI32", adev.gmc.vm_end >> 12, inst=inst)
  bus = WREG32_PAIR(bus, adev, f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_BASE_ADDR", "_LO32", "_HI32", adev.paddr2xgmi(page_table.paddr) | 1, inst=inst)

  fault_flags = {f'{x}_protection_fault_enable_interrupt':1 for x in ['pde0', 'dummy_page', 'range', 'valid', 'read', 'write', 'execute']}
  en_def_flags = {f'{x}_protection_fault_enable_default':1 for x in ['pde0', 'dummy_page', 'range', 'valid', 'read', 'write', 'execute']}
  return WREG32(bus, adev.reg(f"reg{ip}VM_CONTEXT{vmid}_CNTL"), 0x1800000, **fault_flags, **en_def_flags, enable_context=1,
    page_table_depth=((2 if adev.gmc.trans_futher else 3) - page_table.lv), page_table_block_size=9 if adev.gmc.trans_futher else 0, inst=inst)

@uopfunc
def gmc_init_hub(bus:UOp, adev, ip:Literal["MM", "GC"], insts:Iterable[int]) -> UOp:
  # Init system apertures
  for inst in insts:
    bus = WREG32(bus, adev.reg(f"reg{ip}MC_VM_AGP_BASE"), 0, inst=inst)
    bus = WREG32(bus, adev.reg(f"reg{ip}MC_VM_AGP_BOT"), 0xffffffffffff >> 24, inst=inst) # disable AGP
    bus = WREG32(bus, adev.reg(f"reg{ip}MC_VM_AGP_TOP"), 0, inst=inst)

    bus = WREG32(bus, adev.reg(f"reg{ip}MC_VM_SYSTEM_APERTURE_LOW_ADDR"), adev.gmc.fb_base >> 18, inst=inst)
    bus = WREG32(bus, adev.reg(f"reg{ip}MC_VM_SYSTEM_APERTURE_HIGH_ADDR"), adev.gmc.fb_end >> 18, inst=inst)
    bus = WREG32_PAIR(bus, adev, f"reg{ip}MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR", "_LSB", "_MSB", adev.gmc.memscratch_xgmi_paddr >> 12, inst=inst)
    bus = WREG32_PAIR(bus, adev, f"reg{ip}VM_L2_PROTECTION_FAULT_DEFAULT_ADDR", "_LO32", "_HI32", adev.gmc.dummy_page_xgmi_paddr >> 12, inst=inst)

    bus = WREG32_FIELD(bus, adev.reg(f"reg{ip}VM_L2_PROTECTION_FAULT_CNTL2"), active_page_migration_pte_read_retry=1, inst=inst)

    # Init TLB and cache
    bus = WREG32_FIELD(bus, adev.reg(f"reg{ip}MC_VM_MX_L1_TLB_CNTL"), enable_l1_tlb=1, system_access_mode=3, enable_advanced_driver_model=1,
      system_aperture_unmapped_access=0, mtype=adev.soc.module.MTYPE_UC, inst=inst)

    bus = WREG32_FIELD(bus, adev.reg(f"reg{ip}VM_L2_CNTL"), enable_l2_cache=1, enable_default_page_out_to_system_memory=1,
      l2_pde0_cache_tag_generation_mode=0, pde_fault_classification=0, context1_identity_access_mode=1, identity_mode_fragment_size=0,
      enable_l2_fragment_processing=int(adev.ip_ver[am.GC_HWIP] < (10,0,0)), inst=inst)
    bus = WREG32_FIELD(bus, adev.reg(f"reg{ip}VM_L2_CNTL2"), invalidate_all_l1_tlbs=1, invalidate_l2_cache=1, inst=inst)
    bus = WREG32(bus, adev.reg(f"reg{ip}VM_L2_CNTL3"), l2_cache_4k_associativity=1, l2_cache_bigk_associativity=1,
      bank_select=12 if adev.gmc.trans_futher else 9, l2_cache_bigk_fragment_size=9 if adev.gmc.trans_futher else 6, inst=inst)
    bus = WREG32(bus, adev.reg(f"reg{ip}VM_L2_CNTL4"), l2_cache_4k_partition_count=1, inst=inst)
    if adev.ip_ver[am.GC_HWIP] >= (10,0,0): bus = WREG32(bus, adev.reg(f"reg{ip}VM_L2_CNTL5"), walker_priority_client_id=0x1ff, inst=inst)

    bus = gmc_enable_vm_addressing(bus, adev, adev.mm.root_page_table, ip, vmid=0, inst=inst)

    # Disable identity aperture
    bus = WREG32_PAIR(bus, adev, f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR", "_LO32", "_HI32", 0xfffffffff, inst=inst)
    bus = WREG32_PAIR(bus, adev, f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR", "_LO32", "_HI32", 0x0, inst=inst)
    bus = WREG32_PAIR(bus, adev, f"reg{ip}VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", "_LO32", "_HI32", 0x0, inst=inst)

    for eng_i in range(18): bus = WREG32_PAIR(bus, adev, f"reg{ip}VM_INVALIDATE_ENG{eng_i}_ADDR_RANGE", "_LO32", "_HI32", 0x1fffffffff, inst=inst)
  adev.gmc.hub_initted[ip] = True
  return bus.sink()

# **** smu

class AM_SMU(AM_IP):
  def init_sw(self):
    self.smu_mod = self.adev._ip_module("smu", am.MP1_HWIP)
    self.driver_table_paddr = self.adev.mm.palloc(0x4000, zero=False, boot=True)

  def read_table(self, table_t, arg):
    if self.adev.ip_ver[am.MP0_HWIP] in {(13,0,6),(13,0,12),(13,0,15)}: self.adev.run(smu_send_msg, self.smu_mod.PPSMC_MSG_GetMetricsTable, arg)
    else: self.adev.run(smu_send_msg, self.smu_mod.PPSMC_MSG_TransferTableSmu2Dram, arg)
    return table_t.from_buffer(bytearray(self.adev.vram.view(self.driver_table_paddr, ctypes.sizeof(table_t))[:]))

  def aca_read_reg(self, bank_idx:int, reg_idx:int, ue=True) -> int:
    msg = self.smu_mod.PPSMC_MSG_McaBankDumpDW if ue else self.smu_mod.PPSMC_MSG_McaBankCeDumpDW
    return (self.adev.run(smu_send_msg, msg, (bank_idx << 16) | (reg_idx * 8 + 4), read_back_arg=True)[1] << 32) | \
            self.adev.run(smu_send_msg, msg, (bank_idx << 16) | (reg_idx * 8), read_back_arg=True)[1]

  def aca_read_banks(self, ue=True) -> list[list[int]]:
    if not hasattr(self.smu_mod, 'PPSMC_MSG_QueryValidMcaCount'): return []
    count_msg = self.smu_mod.PPSMC_MSG_QueryValidMcaCount if ue else self.smu_mod.PPSMC_MSG_QueryValidMcaCeCount
    count = self.adev.run(smu_send_msg, count_msg, 0, read_back_arg=True)[1]
    return [[self.aca_read_reg(idx, reg_idx, ue=ue) for reg_idx in range(16)] for idx in range(count)]

@uopfunc
def smu_init_hw(bus:UOp, adev) -> UOp:
  bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_SetDriverDramAddrHigh, hi32(adev.paddr2mc(adev.smu.driver_table_paddr))))
  bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_SetDriverDramAddrLow, lo32(adev.paddr2mc(adev.smu.driver_table_paddr))))
  bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_EnableAllSmuFeatures, 0))
  return bus.sink()

@uopfunc
def smu_is_smu_alive(bus:UOp, adev) -> UOp:
  bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_GetSmuVersion, 0, timeout=100, suppress=True))
  bus, resp = RREG32(bus, adev.mmMP1_SMN_C2PMSG_90)
  return am_out(bus, 1, resp != 0).sink()

@uopfunc
def smu_mode1_reset(bus:UOp, adev) -> UOp:
  if DEBUG >= 2: print(f"am {adev.devfmt}: mode1 reset")
  if adev.ip_ver[am.MP0_HWIP] >= (14,0,0) or adev.ip_ver[am.MP0_HWIP] in {(13,0,0), (13,0,7), (13,0,10)}:
    bus = bus.after(smu_send_msg(bus, adev, __DEBUGSMC_MSG_Mode1Reset:=2, 0, debug=True))
  elif adev.ip_ver[am.MP0_HWIP] in {(13,0,6), (13,0,12), (13,0,15)}:
    bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_GfxDriverReset, 1))
  else: bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_Mode1Reset, 0))
  return bus.sink()

def smu_read_clocks(bus:UOp, adev, clck:int, level:int) -> tuple[UOp, UOp, UOp]: # the bus, the levels a clock has and its frequency at level
  msg, freq = adev.smu.smu_mod.PPSMC_MSG_GetDpmFreqByIndex, UOp.placeholder((1,), dtypes.uint32, addrspace=AddrSpace.REG)
  bus = bus.after(smu_send_msg(bus, adev, msg, (clck<<16)|0xff))
  bus, cnt = RREG32(bus, adev.mmMP1_SMN_C2PMSG_82)
  i = UOp.range((cnt:=(cnt&0x7fffffff).cast(dtypes.int)), next(UOp.unique_num), dtype=dtypes.int)
  loop = bus.after(i)
  loop = loop.after(smu_send_msg(loop, adev, msg, (clck<<16)|i))
  loop, val = RREG32(loop, adev.mmMP1_SMN_C2PMSG_82)
  stored = freq.after(loop).index(UOp.const(0).valid(i.eq(level if level >= 0 else cnt + level))).store(val&0x7fffffff)
  bus = bus.after(stored.end(i))
  return bus, cnt, freq.after(bus).index(0).load()

@uopfunc
def smu_set_clocks(bus:UOp, adev, level:int|None) -> UOp:
  clks = tuple([adev.smu.smu_mod.PPCLK_UCLK, adev.smu.smu_mod.PPCLK_FCLK, adev.smu.smu_mod.PPCLK_SOCCLK])
  if adev.ip_ver[am.MP0_HWIP] not in {(13,0,6), (13,0,12), (13,0,15)}: clks += (adev.smu.smu_mod.PPCLK_GFXCLK,)

  if level is None:
    for clck in clks:
      bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_SetSoftMinByFreq, clck << 16, timeout=20, suppress=True))
      if adev.ip_ver[am.GC_HWIP] >= (10,0,0):
        bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_SetSoftMaxByFreq, clck << 16 | 0xffff))
    return bus.sink()

  clocks = []
  for clck in clks:
    bus, cnt, val = smu_read_clocks(bus, adev, clck, level)
    clocks.append((clck, cnt, val))

  for clck, cnt, val in clocks:
    live = UOp.range((cnt != 0).cast(dtypes.int), next(UOp.unique_num), dtype=dtypes.int)
    then = bus.after(live)
    then = then.after(smu_send_msg(then, adev, adev.smu.smu_mod.PPSMC_MSG_SetSoftMinByFreq, clck << 16 | val, timeout=20, suppress=True))
    if adev.ip_ver[am.GC_HWIP] >= (10,0,0): then = then.after(smu_send_msg(then, adev, adev.smu.smu_mod.PPSMC_MSG_SetSoftMaxByFreq, clck << 16 | val))
    bus = bus.after(UOp(Ops.NOOP).after(then).end(live))
  return bus.sink()

@uopfunc
def smu_set_power_limit(bus:UOp, adev, watts:float) -> UOp:
  ppt_limit = max(int(round(watts)), 1)
  bus = bus.after(smu_send_msg(bus, adev, adev.smu.smu_mod.PPSMC_MSG_SetPptLimit, ppt_limit))
  if DEBUG >= 2: print(f"am {adev.devfmt}: GPU power limit set to {ppt_limit}W")
  return bus.sink()

def smu_cmn_send_msg(bus:UOp, adev, msg:int, param=0, debug=False) -> UOp:
  bus = WREG32(bus, adev.mmMP1_SMN_C2PMSG_90 if not debug else adev.mmMP1_SMN_C2PMSG_54, 0) # resp reg
  bus = WREG32(bus, adev.mmMP1_SMN_C2PMSG_82 if not debug else adev.mmMP1_SMN_C2PMSG_53, param)
  return WREG32(bus, adev.mmMP1_SMN_C2PMSG_66 if not debug else adev.mmMP1_SMN_C2PMSG_75, msg)

@uopfunc
def smu_send_msg(bus:UOp, adev, msg:int, param:UOp|int, read_back_arg=False, timeout=10000, debug=False, suppress=False) -> UOp:
  bus = smu_cmn_send_msg(bus, adev, msg, param, debug=debug)
  bus = wait_cond(bus, adev, lambda bus: RREG32(bus, adev.mmMP1_SMN_C2PMSG_90 if not debug else adev.mmMP1_SMN_C2PMSG_54), value=1,
    timeout_ms=timeout, msg=None if suppress else f"SMU msg {msg:#x} timeout") # default timeout is 10 seconds
  if read_back_arg:
    bus, arg = RREG32(bus, adev.mmMP1_SMN_C2PMSG_82 if not debug else adev.mmMP1_SMN_C2PMSG_53)
    bus = am_out(bus, 1, arg)
  return bus.sink()

# **** gfx

class AM_GFX(AM_IP):
  def init_sw(self):
    self.xccs = sum(1 for i in self.adev.regs_offset[am.GC_HWIP] if i not in self.adev.harvested[am.GC_HWIP])
    self.mqd_paddr = [self.adev.mm.palloc(0x1000 * self.xccs, zero=False, boot=True) for i in range(2)]
    self.mqd_mc = [self.adev.paddr2mc(mqd_paddr) for mqd_paddr in self.mqd_paddr]

  def setup_ring(self, *args, **kwargs) -> int: return self.adev.run(gfx_setup_ring, *args, **kwargs)[1]

@uopfunc
def gfx_init_hw(bus:UOp, adev) -> UOp:
  # Wait for RLC autoload to complete
  def autoload(bus:UOp) -> tuple[UOp, UOp]:
    bus, cp_stat = RREG32(bus, adev.regCP_STAT)
    if not hasattr(adev, 'regRLC_RLCS_BOOTLOAD_STATUS'): return bus, cp_stat.eq(0)
    bus, bootload = RREG32(bus, adev.regRLC_RLCS_BOOTLOAD_STATUS)
    return bus, cp_stat.eq(0) | (bootload & adev.regRLC_RLCS_BOOTLOAD_STATUS.fields_mask('bootload_complete')).eq(0)
  bus = wait_cond(bus, adev, autoload, value=True, msg="RLC autoload timeout")

  bus = bus.after(gmc_init_hub(bus, adev, "GC", insts=range(adev.gfx.xccs)))
  if adev.partial_boot: return bus.after(gfx_reset_mec(bus, adev)).sink()

  bus = bus.after(gfx_config_mec(bus, adev))

  # NOTE: Golden reg for gfx11. No values for this reg provided. The kernel just ors 0x20000000 to this reg.
  for xcc in range(adev.gfx.xccs):
    bus, tcp_cntl = RREG32(bus, adev.regTCP_CNTL)
    bus = WREG32(bus, adev.regTCP_CNTL, tcp_cntl | 0x20000000, inst=xcc)

  for xcc in range(adev.gfx.xccs): bus = WREG32(bus, adev.regRLC_CNTL, 0x1, inst=xcc)

  for xcc in range(adev.gfx.xccs): bus = WREG32_FIELD(bus, adev.regRLC_SRM_CNTL, srm_enable=1, auto_incr_addr=1, inst=xcc)

  for xcc in range(adev.gfx.xccs): bus = WREG32(bus, adev.regRLC_SPM_MC_CNTL, 0xf, inst=xcc)

  if adev.ip_ver[am.NBIO_HWIP][:2] != (7,9):
    bus = soc_doorbell_enable(bus, adev, port=0, awid=0x3, awaddr_31_28_value=0x3)
    bus = soc_doorbell_enable(bus, adev, port=3, awid=0x6, awaddr_31_28_value=0x3)

  for xcc in range(adev.gfx.xccs):
    if adev.ip_ver[am.GC_HWIP] in {(9,4,3), (9,5,0)}:
      bus = WREG32(bus, adev.regGB_ADDR_CONFIG, 0x2a114042, inst=xcc) # Golden value for mi300/mi350
      bus = WREG32_FIELD(bus, adev.regTCP_UTCL1_CNTL2, spare=1, inst=xcc)

    bus = WREG32_FIELD(bus, adev.regGRBM_CNTL, read_timeout=0xff, inst=xcc)
    i = UOp.range(16, next(UOp.unique_num), dtype=dtypes.int)
    loop = gfx_grbm_select(bus.after(i), adev, vmid=i, inst=xcc)
    loop = WREG32(loop, adev.regSH_MEM_CONFIG, **({'initial_inst_prefetch':3} if adev.ip_ver[am.GC_HWIP][0]>=10 else {'retry_disable':1}),
      **({'f8_mode':1} if adev.ip_ver[am.GC_HWIP][:2]==(9,4) else {}),
      address_mode=adev.soc.module.SH_MEM_ADDRESS_MODE_64, alignment_mode=adev.soc.module.SH_MEM_ALIGNMENT_MODE_UNALIGNED, inst=xcc)

    # Configure apertures:
    # LDS:         0x10000000'00000000 - 0x10000001'00000000 (4GB)
    # Scratch:     0x20000000'00000000 - 0x20000001'00000000 (4GB)
    loop = WREG32(loop, adev.regSH_MEM_BASES, shared_base=0x1, private_base=0x2, inst=xcc)
    bus = gfx_grbm_select(bus.after(UOp(Ops.NOOP).after(loop).end(i)), adev, inst=xcc)

    # Configure MEC doorbell range
    bus = WREG32(bus, adev.regCP_MEC_DOORBELL_RANGE_LOWER, 0x100 * xcc, inst=xcc)
    bus = WREG32(bus, adev.regCP_MEC_DOORBELL_RANGE_UPPER, 0x100 * xcc + 0xf8, inst=xcc)

  bus = bus.after(gfx_enable_mec(bus, adev))

  # set 1 partition on bare metal.
  if adev.gfx.xccs > 1: bus = bus.after(psp_spatial_partition_cmd(bus, adev, 1))
  return bus.sink()

@uopfunc
def gfx_fini_hw(bus:UOp, adev) -> UOp: return bus.after(gfx_dequeue_hqds(bus, adev)).sink()

@uopfunc
def gfx_reset_mec(bus:UOp, adev) -> UOp:
  bus = bus.after(gfx_dequeue_hqds(bus, adev))

  if adev.ip_ver[am.GC_HWIP] < (12,0,0): # gfx12+ uses mec_pipe0_reset
    bus = gfx_rlc_safe_mode(bus, adev, enter=True)
    for xcc in range(adev.gfx.xccs): bus = WREG32(bus, adev.regGRBM_SOFT_RESET, soft_reset_cp=1, soft_reset_cpc=1, inst=xcc)
    bus = sleep(bus, 0.05)
    for xcc in range(adev.gfx.xccs): bus = WREG32(bus, adev.regGRBM_SOFT_RESET, 0x0, inst=xcc)
    bus = gfx_rlc_safe_mode(bus, adev, enter=False)

  bus = bus.after(gfx_config_mec(bus, adev))
  bus = bus.after(gfx_enable_mec(bus, adev))
  return bus.sink()

@uopfunc
def gfx_setup_ring(bus:UOp, adev, ring_addr:int, ring_size:int, rptr_addr:int, wptr_addr:int, eop_addr:int, eop_size:int, idx:int,
                   aql:bool) -> UOp:
  pipe, queue, doorbell = idx // 4, idx % 4, am.AMDGPU_NAVI10_DOORBELL_MEC_RING0

  for xcc in range(adev.gfx.xccs if aql else 1):
    bus = gfx_grbm_select(bus, adev, me=1, pipe=pipe, queue=queue, inst=xcc)

    struct_t = getattr(am, f"struct_v{adev.ip_ver[am.GC_HWIP][0]}{'_compute' if adev.ip_ver[am.GC_HWIP][0] >= 10 else ''}_mqd")
    mqd_struct = struct_t(header=0xC0310800, cp_mqd_base_addr_lo=lo32(adev.gfx.mqd_mc[queue] + 0x1000*xcc),
      cp_mqd_base_addr_hi=hi32(adev.gfx.mqd_mc[queue] + 0x1000*xcc), cp_hqd_pipe_priority=0x2, cp_hqd_queue_priority=0xf,
      cp_hqd_quantum=0x111, cp_hqd_persistent_state=adev.regCP_HQD_PERSISTENT_STATE.encode(preload_size=0x55, preload_req=1),
      cp_hqd_pq_base_lo=lo32(ring_addr>>8), cp_hqd_pq_base_hi=hi32(ring_addr>>8),
      cp_hqd_pq_rptr_report_addr_lo=lo32(rptr_addr), cp_hqd_pq_rptr_report_addr_hi=hi32(rptr_addr),
      cp_hqd_pq_wptr_poll_addr_lo=lo32(wptr_addr), cp_hqd_pq_wptr_poll_addr_hi=hi32(wptr_addr),
      cp_hqd_pq_doorbell_control=adev.regCP_HQD_PQ_DOORBELL_CONTROL.encode(doorbell_offset=doorbell*2, doorbell_en=1),
      cp_hqd_pq_control=adev.regCP_HQD_PQ_CONTROL.encode(rptr_block_size=5, unord_dispatch=0, queue_size=(ring_size//4).bit_length()-2,
        **({'queue_full_en':1, 'slot_based_wptr':2, 'no_update_rptr':xcc!=0 or adev.gfx.xccs==1} if aql else {})),
      cp_hqd_ib_control=adev.regCP_HQD_IB_CONTROL.encode(min_ib_avail_size=0x3), cp_hqd_hq_status0=0x20004000,
      cp_mqd_control=adev.regCP_MQD_CONTROL.encode(priv_state=1), cp_hqd_vmid=0, cp_hqd_aql_control=int(aql),
      cp_hqd_eop_base_addr_lo=lo32(eop_addr>>8), cp_hqd_eop_base_addr_hi=hi32(eop_addr>>8),
      cp_hqd_eop_control=adev.regCP_HQD_EOP_CONTROL.encode(eop_size=(eop_size//4).bit_length()-2),
      **({'compute_tg_chunk_size':1, 'compute_current_logic_xcc_id':xcc, 'cp_mqd_stride_size':0x1000} if aql and adev.gfx.xccs > 1 else {}))
    for se in range(8 if adev.ip_ver[am.GC_HWIP][0] >= 10 else 4): setattr(mqd_struct, f'compute_static_thread_mgmt_se{se}', 0xffffffff)

    bus = am_memcpy(bus, adev, adev.rvram, adev.gfx.mqd_paddr[queue] + 0x1000*xcc, bytes(mqd_struct))

    mqd_st_mv = to_mv(ctypes.addressof(mqd_struct), ctypes.sizeof(mqd_struct)).cast('I')
    for i, reg in enumerate(range(adev.regCP_MQD_BASE_ADDR.addr[xcc], adev.regCP_HQD_PQ_WPTR_HI.addr[xcc] + 1)):
      bus = am_wreg(bus, adev, reg, mqd_st_mv[0x80 + i], inst=xcc)
    bus = WREG32(bus, adev.regCP_HQD_ACTIVE, 0x1, inst=xcc)

    bus = gmc_flush_hdp(bus, adev)
    bus = gfx_grbm_select(bus, adev, inst=xcc)
  return am_out(bus, 1, u64(doorbell)).sink()

@uopfunc
def gfx_set_clockgating_state(bus:UOp, adev) -> UOp:
  if hasattr(adev, 'regMM_ATC_L2_MISC_CG'): bus = WREG32(bus, adev.regMM_ATC_L2_MISC_CG, enable=1, mem_ls_enable=1)

  bus = gfx_rlc_safe_mode(bus, adev, enter=True)
  for xcc in range(adev.gfx.xccs):
    bus = WREG32_FIELD(bus, adev.regRLC_CGCG_CGLS_CTRL, cgcg_gfx_idle_threshold=0x36, cgcg_en=1, cgls_rep_compansat_delay=0xf, cgls_en=1, inst=xcc)

    bus = WREG32_FIELD(bus, adev.regCP_RB_WPTR_POLL_CNTL, poll_frequency=0x100, idle_poll_count=0x90, inst=xcc)
    bus = WREG32_FIELD(bus, adev.regCP_INT_CNTL, cntx_busy_int_enable=1, cntx_empty_int_enable=1, cmp_busy_int_enable=1, inst=xcc)
    if adev.ip_ver[am.GC_HWIP] >= (10,0,0):
      bus = WREG32_FIELD(bus, adev.regSDMA0_RLC_CGCG_CTRL, cgcg_int_enable=1, inst=xcc)
      bus = WREG32_FIELD(bus, adev.regSDMA1_RLC_CGCG_CTRL, cgcg_int_enable=1, inst=xcc)

    feats_gfx9 = {'gfxip_mgls_override':0, 'gfxip_rep_fgcg_override':0} if adev.ip_ver[am.GC_HWIP][0] == 9 else {}
    feats_gfx11 = {'perfmon_clock_state':1, 'gfxip_repeater_fgcg_override':0} if adev.ip_ver[am.GC_HWIP][0] >= 11 else {}
    bus = WREG32_FIELD(bus, adev.regRLC_CGTT_MGCG_OVERRIDE, **feats_gfx9, **feats_gfx11, gfxip_fgcg_override=0, grbm_cgtt_sclk_override=0,
      rlc_cgtt_sclk_override=0, gfxip_mgcg_override=0, gfxip_cgls_override=0, gfxip_cgcg_override=0, inst=xcc)
  bus = gfx_rlc_safe_mode(bus, adev, enter=False)
  return bus.sink()

def gfx_rlc_safe_mode(bus:UOp, adev, enter:bool) -> UOp:
  for xcc in range(adev.gfx.xccs):
    bus = WREG32(bus, adev.regRLC_SAFE_MODE, message=int(enter), cmd=1, inst=xcc)
    if enter: bus = wait_cond(bus, adev, lambda bus: RREG32(bus, adev.regRLC_SAFE_MODE, inst=xcc), mask=0x1, value=0, msg="RLC safe mode timeout")
  return bus

def gfx_grbm_select(bus:UOp, adev, me=0, pipe=0, queue=0, vmid=0, inst=0) -> UOp:
  return WREG32(bus, adev.regGRBM_GFX_CNTL, meid=me, pipeid=pipe, vmid=vmid, queueid=queue, inst=inst)

@uopfunc
def gfx_enable_mec(bus:UOp, adev) -> UOp:
  for xcc in range(adev.gfx.xccs):
    if adev.ip_ver[am.GC_HWIP] >= (10,0,0):
      bus = WREG32_FIELD(bus, adev.regCP_MEC_RS64_CNTL, mec_pipe0_reset=0, mec_pipe0_active=1, mec_halt=0, inst=xcc)
    else: bus = WREG32(bus, adev.regCP_MEC_CNTL, 0x0, inst=xcc)
  return sleep(bus, 0.05).sink()  # Wait for MEC to be ready

@uopfunc
def gfx_halt_engines(bus:UOp, adev) -> UOp:
  for xcc in range(adev.gfx.xccs):
    if adev.ip_ver[am.GC_HWIP] >= (10,0,0): bus = WREG32_FIELD(bus, adev.regCP_MEC_RS64_CNTL, mec_halt=1, inst=xcc)
    else: bus = WREG32_FIELD(bus, adev.regCP_MEC_CNTL, mec_me1_halt=1, mec_me2_halt=1, inst=xcc)
  return bus.sink()

@uopfunc
def gfx_config_mec(bus:UOp, adev) -> UOp:
  def _config_helper(bus:UOp, eng_name, cntl_reg, eng_reg, pipe_cnt, me=0, xcc=0) -> UOp:
    for pipe in range(pipe_cnt):
      bus = gfx_grbm_select(bus, adev, me=me, pipe=pipe, inst=xcc)
      bus = WREG32_PAIR(bus, adev, f"regCP_{eng_reg}_PRGRM_CNTR_START", "", "_HI", adev.fw.ucode_start[eng_name] >> 2, inst=xcc)
    bus = gfx_grbm_select(bus, adev, inst=xcc)
    bus = WREG32_FIELD(bus, adev.reg(f"regCP_{cntl_reg}_CNTL"), **{f"{eng_name.lower()}_pipe{pipe}_reset": 1 for pipe in range(pipe_cnt)}, inst=xcc)
    return WREG32_FIELD(bus, adev.reg(f"regCP_{cntl_reg}_CNTL"), **{f"{eng_name.lower()}_pipe{pipe}_reset": 0 for pipe in range(pipe_cnt)}, inst=xcc)

  for xcc in range(adev.gfx.xccs):
    if adev.ip_ver[am.GC_HWIP] < (10,0,0):
      bus = WREG32_FIELD(bus, adev.regCP_MEC_CNTL, mec_invalidate_icache=1, mec_me1_pipe0_reset=1, mec_me2_pipe0_reset=1, mec_me1_halt=1,
                         mec_me2_halt=1, inst=xcc)
    if adev.ip_ver[am.GC_HWIP] >= (12,0,0):
      bus = _config_helper(bus, eng_name="PFP", cntl_reg="ME", eng_reg="PFP", pipe_cnt=1, xcc=xcc)
      bus = _config_helper(bus, eng_name="ME", cntl_reg="ME", eng_reg="ME", pipe_cnt=1, xcc=xcc)
    if adev.ip_ver[am.GC_HWIP] >= (10,0,0):
      bus = _config_helper(bus, eng_name="MEC", cntl_reg="MEC_RS64", eng_reg="MEC_RS64", pipe_cnt=1, me=1, xcc=xcc)
  return bus.sink()

@uopfunc
def gfx_dequeue_hqds(bus:UOp, adev) -> UOp:
  for me, pipe, q in [(1, 0, 0), (1, 0, 1)]:
    for xcc in range(adev.gfx.xccs):
      bus = gfx_grbm_select(bus, adev, me=me, pipe=pipe, queue=q, inst=xcc)
      bus, active = RREG32(bus, adev.regCP_HQD_ACTIVE, inst=xcc)
      live = UOp.range((active & 1).cast(dtypes.int), next(UOp.unique_num), dtype=dtypes.int)
      then = WREG32(bus.after(live), adev.regCP_HQD_DEQUEUE_REQUEST, 0x2, inst=xcc) # 1 - DRAIN_PIPE; 2 - RESET_WAVES
      then = WREG32(then, adev.regSPI_COMPUTE_QUEUE_RESET, 0x1, inst=xcc)
      if not adev.is_err_state:
        # kernel tolerates this too; a wedged wave can survive RESET_WAVES
        then = wait_cond(then, adev, lambda bus: RREG32(bus, adev.regCP_HQD_ACTIVE, inst=xcc), mask=0x1, value=0, msg=None)
      bus = bus.after(UOp(Ops.NOOP).after(then).end(live))
  bus = gfx_grbm_select(bus, adev)
  return bus.sink()

# **** ih

class AM_IH(AM_IP):
  def init_sw(self):
    self.ring_size = 256 << 10
    def _alloc_ring(size): return (self.adev.mm.palloc(size, zero=False, boot=True), self.adev.mm.palloc(0x1000, zero=False, boot=True))
    self.rings = [(*_alloc_ring(self.ring_size), "", 0), (*_alloc_ring(self.ring_size), "_RING1", 1)]
    self.ring_view = self.adev.vram.view(offset=self.rings[0][0], size=self.ring_size, fmt='I')

  def drain(self):
    _, _, suf, _ = self.rings[0]
    wptr = self.adev.reg(f"regIH_RB_WPTR{suf}").read_bitfields()
    self.adev.regIH_RB_RPTR.write(wptr['offset'] % (self.ring_size // 4))

    if wptr['rb_overflow']:
      self.adev.reg(f"regIH_RB_WPTR{suf}").update(rb_overflow=0)
      self.adev.reg(f"regIH_RB_CNTL{suf}").update(wptr_overflow_clear=1)
      self.adev.reg(f"regIH_RB_CNTL{suf}").update(wptr_overflow_clear=0)

  def interrupt_handler(self):
    _, _, suf, _ = self.rings[0]
    wptr = self.adev.reg(f"regIH_RB_WPTR{suf}").read_bitfields()
    rptr = self.adev.regIH_RB_RPTR.read()

    while rptr != wptr['offset']:
      entry = [self.ring_view[(rptr + i) % (self.ring_size // 4)] for i in range(8)]
      rptr = (rptr + 8) % (self.ring_size // 4)

      client, src, ring_id, vmid, vmid_type, pasid, node = \
        [getattr(am, f'SOC15_{n}_FROM_IH_ENTRY')(entry) for n in ['CLIENT_ID', 'SOURCE_ID', 'RING_ID', 'VMID', 'VMID_TYPE', 'PASID', 'NODEID']]
      ctx = [getattr(am, f'SOC15_CONTEXT_ID{i}_FROM_IH_ENTRY')(entry) for i in range(4)]

      src_name = self.adev.soc.ih_srcs_names.get(client, {}).get(src, '')
      if src_name in {"SDMA_TRAP", "CP_EOP_INTR"}: continue

      print(f"am {self.adev.devfmt}: IH ({rptr:#x}/{wptr['offset']:#x}) client={self.adev.soc.ih_clients.get(client)} src={src_name}({src}) "
            f"ring={ring_id} vmid={vmid}({vmid_type}) pasid={pasid} node={node} ctx=[{ctx[0]:#x}, {ctx[1]:#x}, {ctx[2]:#x}, {ctx[3]:#x}]")

      if src_name == "SQ_INTERRUPT_ID":
        enc_type = getbits(ctx[1], 6, 7) if (is_soc21:=self.adev.ip_ver[am.GC_HWIP][0] >= 11) else getbits(ctx[0], 26, 27)
        err_type = getbits(ctx[0], 21, 24) if is_soc21 else getbits((ctx[0] & 0xfff) | ((ctx[0]>>16) & 0xf000) | ((ctx[1]<<16) & 0xff0000), 20, 23)
        err_info = f" ({['EDC_FUE', 'ILLEGAL_INST', 'MEMVIOL', 'EDC_FED'][err_type]})" if enc_type == 2 else ""
        print(f"am {self.adev.devfmt}: sq_intr: {['auto', 'wave', 'error'][enc_type]}{err_info}")
        self.adev.is_err_state |= enc_type == 2
      elif src_name == "UTCL2_FAULT" or (self.adev.ip_ver[am.GC_HWIP][0] == 9 and client == am.SOC15_IH_CLIENTID_UTCL2):
        bf = self.adev.reg(self.adev.gmc.pf_status_reg('GC')).read_bitfields()
        va = (self.adev.reg('regGCVM_L2_PROTECTION_FAULT_ADDR_HI32').read()<<32) | self.adev.reg('regGCVM_L2_PROTECTION_FAULT_ADDR_LO32').read()
        print(f"am {self.adev.devfmt}: GCVM_L2_PROTECTION_FAULT_STATUS: {bf} {va<<12:#x}")
        self.adev.reg('regGCVM_L2_PROTECTION_FAULT_CNTL').update(clear_protection_fault_status_addr=1)
        self.adev.is_err_state = True
      else: self.adev.is_err_state = True

    self.drain()

    bif_intr = self.adev.regBIF_BX0_BIF_DOORBELL_INT_CNTL.read_bitfields()
    athub_err, cntlr_err = bif_intr['ras_athub_err_event_interrupt_status'], bif_intr['ras_cntlr_interrupt_status']
    if athub_err or cntlr_err:
      print(f"am {self.adev.devfmt}: fatal hardware error detected: {'RAS_ATHUB_ERR_EVENT ' if athub_err else ''}{'RAS_CNTLR' if cntlr_err else ''}")

      acas = self.adev.smu.aca_read_banks(ue=True) + self.adev.smu.aca_read_banks(ue=False)
      for regs in acas:
        acatyp = 'Uncorrectable' if (regs[1] >> 61) & 1 and (regs[1] >> 57) & 1 else 'Correctable'
        hwname = f'{self.adev.hwid_names.get((regs[5] >> 32) & 0xFFF, "")} ({(regs[5] >> 32) & 0xFFF:#03x})'
        print(f"am {self.adev.devfmt}: {acatyp} ACA: {hwname} mcatype={(regs[5] >> 48) & 0xFFFF:#06x} regs=[{', '.join(f'{r:#x}' for r in regs)}]")

      self.adev.regBIF_BX0_BIF_DOORBELL_INT_CNTL.write(ras_cntlr_interrupt_clear=cntlr_err, ras_athub_err_event_interrupt_clear=athub_err)
      self.adev.is_err_state = True

@uopfunc
def ih_init_hw(bus:UOp, adev) -> UOp:
  for ring_vm, rwptr_vm, suf, ring_id in adev.ih.rings:
    bus = WREG32_PAIR(bus, adev, "regIH_RB_BASE", suf, f"_HI{suf}", adev.paddr2mc(ring_vm) >> 8)

    bus = WREG32(bus, adev.reg(f"regIH_RB_CNTL{suf}"), mc_space=4, wptr_overflow_clear=1, rb_size=((adev.ih.ring_size//4)-1).bit_length(),
      mc_snoop=1, mc_ro=0, mc_vmid=0, **({'wptr_overflow_enable': 1, 'rptr_rearm': 1} if ring_id == 0 else {'rb_full_drain_enable': 1}))

    if ring_id == 0: bus = WREG32_PAIR(bus, adev, "regIH_RB_WPTR_ADDR", "_LO", "_HI", adev.paddr2mc(rwptr_vm))

    bus = WREG32(bus, adev.reg(f"regIH_RB_WPTR{suf}"), 0)
    bus = WREG32(bus, adev.reg(f"regIH_RB_RPTR{suf}"), 0)

    bus = WREG32(bus, adev.reg(f"regIH_DOORBELL_RPTR{suf}"), enable=0)

  if adev.ip_ver[am.OSSSYS_HWIP] != (4,4,2):
    bus = WREG32_FIELD(bus, adev.regIH_STORM_CLIENT_LIST_CNTL, client18_is_storm_client=1)
    bus = WREG32_FIELD(bus, adev.regIH_INT_FLOOD_CNTL, flood_cntl_enable=1)
    bus = WREG32_FIELD(bus, adev.regIH_MSI_STORM_CTRL, delay=3)

  # toggle interrupts
  for _, rwptr_vm, suf, ring_id in adev.ih.rings:
    bus = WREG32_FIELD(bus, adev.reg(f"regIH_RB_CNTL{suf}"), rb_enable=1, **({'enable_intr': 1} if ring_id == 0 else {}))
  return bus.sink()

# **** sdma

class AM_SDMA(AM_IP):
  def init_sw(self):
    self.sdma_reginst, self.sdma_name = [], "F32" if self.adev.ip_ver[am.SDMA0_HWIP] < (7,0,0) else "MCU"
    self.engs = 16 if self.adev.ip_ver[am.SDMA0_HWIP] < (5,0,0) else \
      sum(len(self.adev.regs_offset[hw].keys() - self.adev.harvested[hw]) for hw in (am.SDMA0_HWIP, am.SDMA1_HWIP))

  def setup_ring(self, *args, **kwargs) -> int: return self.adev.run(sdma_setup_ring, *args, **kwargs)[1]

@uopfunc
def sdma_halt_engines(bus:UOp, adev) -> UOp:
  if adev.ip_ver[am.SDMA0_HWIP] >= (6,0,0):
    for pipe in range(adev.sdma.engs): bus = WREG32_FIELD(bus, adev.reg(f"regSDMA{pipe}_{adev.sdma.sdma_name}_CNTL"), halt=1)
  return bus.sink()

@uopfunc
def sdma_init_hw(bus:UOp, adev) -> UOp:
  for pipe_id in range(adev.sdma.engs):
    pipe, inst = ("", pipe_id) if adev.ip_ver[am.SDMA0_HWIP] < (5,0,0) else (str(pipe_id), 0)

    if adev.ip_ver[am.SDMA0_HWIP] >= (6,0,0):
      bus = WREG32_FIELD(bus, adev.reg(f"regSDMA{pipe}_WATCHDOG_CNTL"), queue_hang_count=100, inst=inst) # 10s, 100ms per unit
      bus = WREG32_FIELD(bus, adev.reg(f"regSDMA{pipe}_UTCL1_CNTL"), resp_mode=3, redo_delay=9, inst=inst)

      # rd=noa, wr=bypass
      bus = WREG32_FIELD(bus, adev.reg(f"regSDMA{pipe}_UTCL1_PAGE"), rd_l2_policy=2, wr_l2_policy=3,
                         **({'llc_noalloc':1} if adev.sdma.sdma_name == "F32" else {}), inst=inst)
      bus = WREG32_FIELD(bus, adev.reg(f"regSDMA{pipe}_{adev.sdma.sdma_name}_CNTL"), halt=0,
                         **{f"{'th1_' if adev.sdma.sdma_name == 'F32' else ''}reset":0}, inst=inst)

    bus = WREG32_FIELD(bus, adev.reg(f"regSDMA{pipe}_CNTL"), trap_enable=1,
      **({'utc_l1_enable':1} if adev.ip_ver[am.SDMA0_HWIP] <= (5,2,0) else {}), inst=inst)

  if adev.ip_ver[am.NBIO_HWIP] in {(7,9,0), (7,9,1)}:
    for aid_id in adev.aids:
      for dev_inst, (port, awid, offset, awaddr) in enumerate([(1, 0xe, 0xe, 0x1), (2, 0x8, 0x8, 0x2), (5, 0x9, 0x9, 0x8), (6, 0xa, 0xa, 0x9)]):
        entry = dev_inst + 1 + 4 * aid_id
        bus = WREG32(bus, adev.reg(f"regDOORBELL0_CTRL_ENTRY_{entry}"), **{f"bif_doorbell{entry}_range_size_entry": 20,
          f"bif_doorbell{entry}_range_offset_entry": (am.AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 + (entry - 1) * 0xA) * 2})
        bus = soc_doorbell_enable(bus, adev, port=port, awid=awid, awaddr_31_28_value=awaddr, offset=offset, size=4, aid=aid_id)
  else:
    bus = soc_doorbell_enable(bus, adev, port=2, awid=0xe, awaddr_31_28_value=0x3, offset=am.AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0*2,
                              size=20*adev.sdma.engs)
  return bus.sink()

@uopfunc
def sdma_fini_hw(bus:UOp, adev) -> UOp:
  for reg, inst in adev.sdma.sdma_reginst:
    bus = WREG32_FIELD(bus, adev.reg(f"{reg}_RB_CNTL"), rb_enable=0, inst=inst)
    bus = WREG32_FIELD(bus, adev.reg(f"{reg}_IB_CNTL"), ib_enable=0, inst=inst)
    bus = WREG32_FIELD(bus, adev.reg(f"{reg}_DOORBELL"), enable=0, inst=inst)
    bus = WREG32_FIELD(bus, adev.reg(f"{reg}_DOORBELL_OFFSET"), offset=0, inst=inst)

  if adev.ip_ver[am.SDMA0_HWIP] >= (6,0,0):
    bus = WREG32(bus, adev.regGRBM_SOFT_RESET, **{f"soft_reset_sdma{pipe}":1 for pipe in range(adev.sdma.engs)})
    bus = sleep(bus, 0.01)
    bus = WREG32(bus, adev.regGRBM_SOFT_RESET, 0x0)
  return bus.sink()

@uopfunc
def sdma_setup_ring(bus:UOp, adev, ring_addr:int, ring_size:int, rptr_addr:int, wptr_addr:int, idx:int) -> UOp:
  if idx >= adev.sdma.engs: raise RuntimeError(f"am {adev.devfmt}: sdma queue {idx} is not available")

  pipe, queue = (idx // 4, idx % 4) if adev.ip_ver[am.SDMA0_HWIP] < (5,0,0) else (idx, 0)
  reg, inst = ("regSDMA_GFX", pipe+queue*4) if adev.ip_ver[am.SDMA0_HWIP][:2] == (4,4) else (f"regSDMA{pipe}_QUEUE{queue}", 0)
  doorbell = am.AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0 + (pipe+queue*4) * 0xA
  adev.sdma.sdma_reginst.append((reg, inst))

  bus = WREG32(bus, adev.reg(f"{reg}_MINOR_PTR_UPDATE"), 0x1, inst=inst)
  bus = WREG32_PAIR(bus, adev, f"{reg}_RB_RPTR", "", "_HI", 0, inst=inst)
  bus = WREG32_PAIR(bus, adev, f"{reg}_RB_WPTR", "", "_HI", 0, inst=inst)
  bus = WREG32_PAIR(bus, adev, f"{reg}_RB_BASE", "", "_HI", ring_addr >> 8, inst=inst)
  bus = WREG32_PAIR(bus, adev, f"{reg}_RB_RPTR_ADDR", "_LO", "_HI", rptr_addr, inst=inst)
  bus = WREG32_PAIR(bus, adev, f"{reg}_RB_WPTR_POLL_ADDR", "_LO", "_HI", wptr_addr, inst=inst)
  bus = WREG32_FIELD(bus, adev.reg(f"{reg}_DOORBELL_OFFSET"), offset=doorbell * 2, inst=inst)
  bus = WREG32_FIELD(bus, adev.reg(f"{reg}_DOORBELL"), enable=1, inst=inst)
  bus = WREG32(bus, adev.reg(f"{reg}_MINOR_PTR_UPDATE"), 0x0, inst=inst)
  bus = WREG32(bus, adev.reg(f"{reg}_RB_CNTL"),
    **({f'{adev.sdma.sdma_name.lower()}_wptr_poll_enable':1} if adev.ip_ver[am.SDMA0_HWIP][:2]!=(4,4) else {}),
    rb_vmid=0, rptr_writeback_enable=1, rptr_writeback_timer=4, rb_enable=1, rb_priv=1, rb_size=(ring_size//4).bit_length()-1, inst=inst)
  bus = WREG32_FIELD(bus, adev.reg(f"{reg}_IB_CNTL"), ib_enable=1, inst=inst)
  return am_out(bus, 1, u64(doorbell)).sink()

# **** psp

class AM_PSP(AM_IP):
  def init_sw(self):
    self.reg_pref = "regMP0_SMN_C2PMSG" if self.adev.ip_ver[am.MP0_HWIP] < (14,0,0) else "regMPASP_SMN_C2PMSG"

    self.msg1_paddr = self.adev.mm.palloc(am.PSP_1_MEG, align=am.PSP_1_MEG, zero=False, boot=True)
    self.msg1_addr, self.msg1_view = self.adev.paddr2mc(self.msg1_paddr), self.adev.vram.view(self.msg1_paddr, am.PSP_1_MEG, 'B')

    self.cmd_paddr = self.adev.mm.palloc(am.PSP_CMD_BUFFER_SIZE, zero=False, boot=True)
    self.fence_paddr = self.adev.mm.palloc(am.PSP_FENCE_BUFFER_SIZE, zero=True, boot=True)

    self.ring_size = 0x10000
    self.ring_paddr = self.adev.mm.palloc(self.ring_size, zero=False, boot=True)

    self.tmr_size, self.tmr_paddr = 0, 0
    self.boot_time_tmr = self.adev.ip_ver[am.MP0_HWIP] in {(13,0,6), (13,0,14), (14,0,2), (14,0,3)}
    self.autoload_tmr = self.adev.ip_ver[am.MP0_HWIP] not in {(13,0,6), (13,0,14)}

    # First runtime allocation on both full and partial boots, so the resident TMR keeps the same address.
    if not self.boot_time_tmr: self.tmr_paddr = self.adev.mm.pa_allocator.alloc(am.PSP_TMR_ALIGNMENT, am.PSP_TMR_ALIGNMENT)

  def tmr_init(self, tmr_size:int): # the size is known after boot: the allocation gets it, at the same address
    self.tmr_size = tmr_size
    if self.boot_time_tmr: return
    self.adev.mm.pa_allocator.free(self.tmr_paddr)
    tmr_paddr = self.adev.mm.pa_allocator.alloc(self.tmr_size, am.PSP_TMR_ALIGNMENT)
    assert tmr_paddr == self.tmr_paddr, f"TMR moved from {self.tmr_paddr:#x} to {tmr_paddr:#x}"

  def is_sos_alive(self): return self.adev.reg(f"{self.reg_pref}_81").read() != 0x0

@uopfunc
def psp_init_hw(bus:UOp, adev) -> UOp:
  spl_key = am.PSP_FW_TYPE_PSP_SPL if adev.ip_ver[am.MP0_HWIP] >= (14,0,0) else am.PSP_FW_TYPE_PSP_KDB
  sos_components = [(am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_KEY_DATABASE), (spl_key, am.PSP_BL__LOAD_TOS_SPL_TABLE),
    (am.PSP_FW_TYPE_PSP_SYS_DRV, am.PSP_BL__LOAD_SYSDRV), (am.PSP_FW_TYPE_PSP_SOC_DRV, am.PSP_BL__LOAD_SOCDRV),
    (am.PSP_FW_TYPE_PSP_INTF_DRV, am.PSP_BL__LOAD_INTFDRV), (am.PSP_FW_TYPE_PSP_DBG_DRV, am.PSP_BL__LOAD_DBGDRV),
    (am.PSP_FW_TYPE_PSP_RAS_DRV, am.PSP_BL__LOAD_RASDRV), (am.PSP_FW_TYPE_PSP_SOS, am.PSP_BL__LOAD_SOSDRV)]

  if not adev.psp.is_sos_alive():
    for fw, compid in sos_components: bus = bus.after(psp_bootloader_load_component(bus, adev, fw, compid))
    bus = wait_cond(bus, adev, lambda bus: psp_is_sos_alive(bus, adev), value=True, msg="sOS failed to start")

  bus = bus.after(psp_ring_create(bus, adev))
  if am.PSP_FW_TYPE_PSP_TOC in adev.fw.sos_fw: bus = bus.after(psp_tmr_init(bus, adev))

  # SMU fw should be loaded before TMR.
  if hasattr(adev.fw, 'smu_psp_desc'): bus = bus.after(psp_load_ip_fw_cmd(bus, adev, *adev.fw.smu_psp_desc))
  if not adev.psp.boot_time_tmr or not adev.psp.autoload_tmr: bus = bus.after(psp_tmr_load_cmd(bus, adev))

  for psp_desc in adev.fw.descs: bus = bus.after(psp_load_ip_fw_cmd(bus, adev, *psp_desc))

  if adev.ip_ver[am.GC_HWIP] >= (11,0,0): bus = bus.after(psp_rlc_autoload_cmd(bus, adev))
  else: bus = bus.after(psp_load_ip_fw_cmd(bus, adev, [am.GFX_FW_TYPE_REG_LIST], adev.fw.sos_fw[am.PSP_FW_TYPE_PSP_RL]))
  return bus.sink()

def psp_is_sos_alive(bus:UOp, adev) -> tuple[UOp, UOp]:
  bus, sos = RREG32(bus, adev.reg(f"{adev.psp.reg_pref}_81"))
  return bus, sos != 0x0

def psp_wait_for_bootloader(bus:UOp, adev) -> UOp:
  return wait_cond(bus, adev, lambda bus: RREG32(bus, adev.reg(f"{adev.psp.reg_pref}_35")), mask=0x80000000, value=0x80000000, msg="BL not ready")

@uopfunc
def psp_prep_msg1(bus:UOp, adev, data:memoryview) -> UOp:
  assert len(data) <= adev.psp.msg1_view.nbytes, f"msg1 buffer is too small {len(data):#x} > {adev.psp.msg1_view.nbytes:#x}"
  padded_data = pad_bytes(bytes(data) + b'\x00' * 4, 16) # HACK: apple's memcpy requires 16-bytes alignment
  bus = am_memcpy(bus, adev, adev.rvram, adev.psp.msg1_paddr, padded_data)
  bus = gmc_flush_hdp(bus, adev)
  return bus.sink()

@uopfunc
def psp_bootloader_load_component(bus:UOp, adev, fw:int, compid:int) -> UOp:
  if fw not in adev.fw.sos_fw: return bus.sink()

  bus = psp_wait_for_bootloader(bus, adev)

  if DEBUG >= 2: print(f"am {adev.devfmt}: loading sos component: {am.enum_psp_fw_type.get(fw)}")

  bus = bus.after(psp_prep_msg1(bus, adev, adev.fw.sos_fw[fw]))
  bus = WREG32(bus, adev.reg(f"{adev.psp.reg_pref}_36"), adev.psp.msg1_addr >> 20)
  bus = WREG32(bus, adev.reg(f"{adev.psp.reg_pref}_35"), compid)

  return (psp_wait_for_bootloader(bus, adev) if compid != am.PSP_BL__LOAD_SOSDRV else bus).sink()

@uopfunc
def psp_tmr_init(bus:UOp, adev) -> UOp:
  if adev.partial_boot: tmr_size = u32(adev.reg("regSCRATCH_REG5").read())
  else:
    # Load TOC and calculate TMR size
    bus = bus.after(psp_prep_msg1(bus, adev, fwm:=adev.fw.sos_fw[am.PSP_FW_TYPE_PSP_TOC]))
    bus, tmr_size = psp_load_toc_cmd(bus, adev, len(fwm))
  return am_out(bus, 2, tmr_size).sink()

@uopfunc
def psp_ring_create(bus:UOp, adev) -> UOp:
  # If the ring is already created, destroy it
  bus, ring = RREG32(bus, adev.reg(f"{adev.psp.reg_pref}_71"))
  live = UOp.range((ring != 0).cast(dtypes.int), next(UOp.unique_num), dtype=dtypes.int)
  then = WREG32(bus.after(live), adev.reg(f"{adev.psp.reg_pref}_64"), am.GFX_CTRL_CMD_ID_DESTROY_RINGS)

  # There might be handshake issue with hardware which needs delay
  then = sleep(then, 0.02)
  bus = bus.after(UOp(Ops.NOOP).after(then).end(live))

  # Wait until the sOS is ready
  bus = wait_cond(bus, adev, lambda bus: RREG32(bus, adev.reg(f"{adev.psp.reg_pref}_64")), mask=0x80000000, value=0x80000000, msg="sOS not ready")

  bus = WREG32_PAIR(bus, adev, adev.psp.reg_pref, "_69", "_70", adev.paddr2mc(adev.psp.ring_paddr))
  bus = WREG32(bus, adev.reg(f"{adev.psp.reg_pref}_71"), adev.psp.ring_size)
  bus = WREG32(bus, adev.reg(f"{adev.psp.reg_pref}_64"), am.PSP_RING_TYPE__KM << 16)

  # There might be handshake issue with hardware which needs delay
  bus = sleep(bus, 0.02)

  return wait_cond(bus, adev, lambda bus: RREG32(bus, adev.reg(f"{adev.psp.reg_pref}_64")), mask=0x8000FFFF, value=0x80000000,
                  msg="sOS ring not created").sink()

def psp_ring_submit(bus:UOp, adev, cmd:am.struct_psp_gfx_cmd_resp, rows=()) -> UOp: # rows: words of cmd the program fills in
  bus, prev_wptr = RREG32(bus, adev.reg(f"{adev.psp.reg_pref}_67"))
  msg = am.struct_psp_gfx_rb_frame(fence_value=0,
    cmd_buf_addr_lo=lo32(adev.paddr2mc(adev.psp.cmd_paddr)), cmd_buf_addr_hi=hi32(adev.paddr2mc(adev.psp.cmd_paddr)),
    fence_addr_lo=lo32(adev.paddr2mc(adev.psp.fence_paddr)), fence_addr_hi=hi32(adev.paddr2mc(adev.psp.fence_paddr)))
  fence_value = [(am.struct_psp_gfx_rb_frame.fence_value.offset, prev_wptr + 1)]

  bus = am_memcpy(bus, adev, adev.rvram, adev.psp.cmd_paddr, bytes(cmd), rows)
  bus = am_memcpy(bus, adev, adev.rvram, adev.psp.ring_paddr + i64(prev_wptr) * 4, bytes(msg), fence_value)

  # Move the wptr
  bus = WREG32(bus, adev.reg(f"{adev.psp.reg_pref}_67"), prev_wptr + ctypes.sizeof(am.struct_psp_gfx_rb_frame) // 4)

  bus = wait_cond(bus, adev, lambda bus: am_load(bus, adev.rvram, adev.psp.fence_paddr), value=prev_wptr + 1,
                  msg="sOS ring not responding")

  resp = adev.psp.cmd_paddr + am.struct_psp_gfx_cmd_resp.resp.offset
  bus, status = am_load(bus, adev.rvram, resp + am.struct_psp_gfx_resp.status.offset)
  return am_fail(bus, adev, status != 0, RuntimeError(f"PSP command failed {cmd.cmd_id}"))

@uopfunc
def psp_load_ip_fw_cmd(bus:UOp, adev, fw_types:list[int], fw_bytes:memoryview) -> UOp:
  bus = bus.after(psp_prep_msg1(bus, adev, fw_bytes))
  for fw_type in fw_types:
    if DEBUG >= 2: print(f"am {adev.devfmt}: loading fw: {am.enum_psp_gfx_fw_type.get(fw_type)}")
    cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_LOAD_IP_FW)
    cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi, cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = data64(adev.psp.msg1_addr)
    cmd.cmd.cmd_load_ip_fw.fw_size = len(fw_bytes)
    cmd.cmd.cmd_load_ip_fw.fw_type = fw_type
    bus = psp_ring_submit(bus, adev, cmd)
  return bus.sink()

@uopfunc
def psp_tmr_load_cmd(bus:UOp, adev) -> UOp:
  tmr_paddr = adev.paddr2xgmi(adev.psp.tmr_paddr) if adev.psp.tmr_paddr else 0

  cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_SETUP_TMR)
  buf_phy_addr = adev.paddr2mc(adev.psp.tmr_paddr) if adev.psp.tmr_paddr else 0
  cmd.cmd.cmd_setup_tmr.buf_phy_addr_hi, cmd.cmd.cmd_setup_tmr.buf_phy_addr_lo = data64(buf_phy_addr)
  cmd.cmd.cmd_setup_tmr.system_phy_addr_hi, cmd.cmd.cmd_setup_tmr.system_phy_addr_lo = data64(tmr_paddr)
  cmd.cmd.cmd_setup_tmr.bitfield.virt_phy_addr = 1
  buf_size = am.struct_psp_gfx_cmd_resp.cmd.offset + am.struct_psp_gfx_cmd_setup_tmr.buf_size.offset
  return psp_ring_submit(bus, adev, cmd, [(buf_size, bus.index(8 + 2).load().cast(dtypes.uint32))] if adev.psp.tmr_paddr else []).sink()

def psp_load_toc_cmd(bus:UOp, adev, toc_size:int) -> tuple[UOp, UOp]:
  cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_LOAD_TOC)
  cmd.cmd.cmd_load_toc.toc_phy_addr_hi, cmd.cmd.cmd_load_toc.toc_phy_addr_lo = data64(adev.psp.msg1_addr)
  cmd.cmd.cmd_load_toc.toc_size = toc_size
  bus = psp_ring_submit(bus, adev, cmd)
  resp = adev.psp.cmd_paddr + am.struct_psp_gfx_cmd_resp.resp.offset
  return am_load(bus, adev.rvram, resp + am.struct_psp_gfx_resp.tmr_size.offset)

@uopfunc
def psp_spatial_partition_cmd(bus:UOp, adev, mode) -> UOp:
  cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_SRIOV_SPATIAL_PART)
  cmd.cmd.cmd_spatial_part.mode = mode
  return psp_ring_submit(bus, adev, cmd).sink()

@uopfunc
def psp_rlc_autoload_cmd(bus:UOp, adev) -> UOp:
  return psp_ring_submit(bus, adev, am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_AUTOLOAD_RLC)).sink()

# **** driver

@uopfunc
def am_reset(bus:UOp, adev) -> UOp:
  # Quiesce first: mode1 reset over live engines can wedge the GPU until power cycled.
  for quiesce in [gfx_fini_hw, gfx_halt_engines, sdma_halt_engines]: bus = bus.after(quiesce(bus, adev))
  bus = sleep(bus, 0.1)
  return bus.after(smu_mode1_reset(bus, adev)).sink()

@uopfunc
def am_boot(bus:UOp, adev) -> UOp:
  # Init hw for IP blocks where it is needed
  if not adev.partial_boot:
    for init_hw in [soc_init_hw, gmc_init_hw, ih_init_hw, psp_init_hw, smu_init_hw]: bus = bus.after(init_hw(bus, adev))
  else: bus = bus.after(psp_tmr_init(bus, adev))

  # Re-initialize main blocks
  for init_hw in [gfx_init_hw, sdma_init_hw]: bus = bus.after(init_hw(bus, adev))

  if (max_power:=getenv("AM_POWER_LIMIT", 0.0)) > 0: bus = bus.after(smu_set_power_limit(bus, adev, max_power))
  levels, perf = {"auto": None, "high": -1, "low": 0}, getenv("AM_PERF_LEVEL", "auto" if max_power > 0 else "high")
  assert perf in levels, f"AM_PERF_LEVEL must be one of {list(levels)}, got {perf!r}"
  bus = bus.after(smu_set_clocks(bus, adev, level=levels[perf]))
  for set_clockgating_state in [soc_set_clockgating_state, gfx_set_clockgating_state]: bus = bus.after(set_clockgating_state(bus, adev))
  # scratch registers are writable after GFX initialization
  bus = WREG32(bus, adev.reg("regSCRATCH_REG5"), bus.index(8 + 2).load().cast(dtypes.uint32))
  bus = WREG32(bus, adev.reg("regSCRATCH_REG7"), adev.Version)
  bus = WREG32(bus, adev.reg("regSCRATCH_REG6"), 1) # set initialized state.
  return bus.sink()

@uopfunc
def am_fini(bus:UOp, adev) -> UOp:
  for fini_hw in [sdma_fini_hw, gfx_fini_hw]: bus = bus.after(fini_hw(bus, adev))
  return bus.after(smu_set_clocks(bus, adev, level=0)).sink()

@uopfunc
def am_recover(bus:UOp, adev) -> UOp:
  for recover in [gfx_reset_mec, sdma_fini_hw, sdma_init_hw]: bus = bus.after(recover(bus, adev))
  return bus.sink()
