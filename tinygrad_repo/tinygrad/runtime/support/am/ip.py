import ctypes, time, contextlib, importlib, functools
from typing import Literal
from tinygrad.runtime.autogen.am import am
from tinygrad.helpers import to_mv, data64, lo32, hi32, DEBUG

class AM_IP:
  def __init__(self, adev): self.adev = adev
  def init_sw(self): pass # Prepare sw/allocations for this IP
  def init_hw(self): pass # Initialize hw for this IP
  def fini_hw(self): pass # Finalize hw for this IP
  def set_clockgating_state(self): pass # Set clockgating state for this IP

class AM_SOC(AM_IP):
  def init_sw(self):
    self.soc_ver = 24 if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else 21
    self.module = importlib.import_module(f"tinygrad.runtime.autogen.am.soc{self.soc_ver}")

  def init_hw(self):
    self.adev.regRCC_DEV0_EPF2_STRAP2.update(strap_no_soft_reset_dev0_f2=0x0)
    self.adev.regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN.write(0x1)
  def set_clockgating_state(self): self.adev.regHDP_MEM_POWER_CTRL.update(atomic_mem_power_ctrl_en=1, atomic_mem_power_ds_en=1)

  def doorbell_enable(self, port, awid=0, awaddr_31_28_value=0, offset=0, size=0):
    self.adev.reg(f"{'regGDC_S2A0_S2A' if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else 'regS2A'}_DOORBELL_ENTRY_{port}_CTRL").update(
      **{f"s2a_doorbell_port{port}_enable":1, f"s2a_doorbell_port{port}_awid":awid, f"s2a_doorbell_port{port}_awaddr_31_28_value":awaddr_31_28_value,
         f"s2a_doorbell_port{port}_range_offset":offset, f"s2a_doorbell_port{port}_range_size":size})

class AM_GMC(AM_IP):
  def init_sw(self):
    # Memory controller aperture
    self.mc_base = (self.adev.regMMMC_VM_FB_LOCATION_BASE.read() & 0xFFFFFF) << 24
    self.mc_end = self.mc_base + self.adev.mm.vram_size - 1

    # VM aperture
    self.vm_base = self.adev.mm.va_allocator.base
    self.vm_end = self.vm_base + self.adev.mm.va_allocator.size - 1

    # GFX11/GFX12 has 44-bit address space
    self.address_space_mask = (1 << 44) - 1

    self.memscratch_paddr = self.adev.mm.palloc(0x1000, zero=not self.adev.partial_boot, boot=True)
    self.dummy_page_paddr = self.adev.mm.palloc(0x1000, zero=not self.adev.partial_boot, boot=True)
    self.hub_initted = {"MM": False, "GC": False}

  def init_hw(self): self.init_hub("MM")

  def flush_hdp(self): self.adev.wreg(self.adev.reg("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL").read() // 4, 0x0)
  def flush_tlb(self, ip:Literal["MM", "GC"], vmid, flush_type=0):
    self.flush_hdp()

    # Can't issue TLB invalidation if the hub isn't initialized.
    if not self.hub_initted[ip]: return

    if ip == "MM": self.adev.wait_reg(self.adev.regMMVM_INVALIDATE_ENG17_SEM, mask=0x1, value=0x1)

    self.adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_REQ").write(flush_type=flush_type, per_vmid_invalidate_req=(1 << vmid), invalidate_l2_ptes=1,
      invalidate_l2_pde0=1, invalidate_l2_pde1=1, invalidate_l2_pde2=1, invalidate_l1_ptes=1, clear_protection_fault_status_addr=0)

    self.adev.wait_reg(self.adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_ACK"), mask=(1 << vmid), value=(1 << vmid))

    if ip == "MM":
      self.adev.regMMVM_INVALIDATE_ENG17_SEM.write(0x0)
      self.adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2.update(reserved_cache_private_invalidation=1)

      # Read back the register to ensure the invalidation is complete
      self.adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2.read()

  def enable_vm_addressing(self, page_table, ip:Literal["MM", "GC"], vmid):
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_START_ADDR", "_LO32", "_HI32", self.vm_base >> 12)
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_END_ADDR", "_LO32", "_HI32", self.vm_end >> 12)
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_BASE_ADDR", "_LO32", "_HI32", page_table.paddr | 1)
    self.adev.reg(f"reg{ip}VM_CONTEXT{vmid}_CNTL").write(0x1800000, pde0_protection_fault_enable_interrupt=1, pde0_protection_fault_enable_default=1,
                                                         dummy_page_protection_fault_enable_interrupt=1, dummy_page_protection_fault_enable_default=1,
                                                         range_protection_fault_enable_interrupt=1, range_protection_fault_enable_default=1,
                                                         valid_protection_fault_enable_interrupt=1, valid_protection_fault_enable_default=1,
                                                         read_protection_fault_enable_interrupt=1, read_protection_fault_enable_default=1,
                                                         write_protection_fault_enable_interrupt=1, write_protection_fault_enable_default=1,
                                                         execute_protection_fault_enable_interrupt=1, execute_protection_fault_enable_default=1,
                                                         enable_context=1, page_table_depth=(3 - page_table.lv))

  def init_hub(self, ip:Literal["MM", "GC"]):
    # Init system apertures
    self.adev.reg(f"reg{ip}MC_VM_AGP_BASE").write(0)
    self.adev.reg(f"reg{ip}MC_VM_AGP_BOT").write(0xffffffffffff >> 24) # disable AGP
    self.adev.reg(f"reg{ip}MC_VM_AGP_TOP").write(0)

    self.adev.reg(f"reg{ip}MC_VM_SYSTEM_APERTURE_LOW_ADDR").write(self.mc_base >> 18)
    self.adev.reg(f"reg{ip}MC_VM_SYSTEM_APERTURE_HIGH_ADDR").write(self.mc_end >> 18)
    self.adev.wreg_pair(f"reg{ip}MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR", "_LSB", "_MSB", self.memscratch_paddr >> 12)
    self.adev.wreg_pair(f"reg{ip}VM_L2_PROTECTION_FAULT_DEFAULT_ADDR", "_LO32", "_HI32", self.dummy_page_paddr >> 12)

    self.adev.reg(f"reg{ip}VM_L2_PROTECTION_FAULT_CNTL2").update(active_page_migration_pte_read_retry=1)

    # Init TLB and cache
    self.adev.reg(f"reg{ip}MC_VM_MX_L1_TLB_CNTL").update(enable_l1_tlb=1, system_access_mode=3, enable_advanced_driver_model=1,
                                                         system_aperture_unmapped_access=0, eco_bits=0, mtype=self.adev.soc.module.MTYPE_UC)

    self.adev.reg(f"reg{ip}VM_L2_CNTL").update(enable_l2_cache=1, enable_l2_fragment_processing=0, enable_default_page_out_to_system_memory=1,
      l2_pde0_cache_tag_generation_mode=0, pde_fault_classification=0, context1_identity_access_mode=1, identity_mode_fragment_size=0)
    self.adev.reg(f"reg{ip}VM_L2_CNTL2").update(invalidate_all_l1_tlbs=1, invalidate_l2_cache=1)
    self.adev.reg(f"reg{ip}VM_L2_CNTL3").write(bank_select=9, l2_cache_bigk_fragment_size=6,l2_cache_4k_associativity=1,l2_cache_bigk_associativity=1)
    self.adev.reg(f"reg{ip}VM_L2_CNTL4").write(l2_cache_4k_partition_count=1)
    self.adev.reg(f"reg{ip}VM_L2_CNTL5").write(walker_priority_client_id=0x1ff)

    self.enable_vm_addressing(self.adev.mm.root_page_table, ip, vmid=0)

    # Disable identity aperture
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR", "_LO32", "_HI32", 0xfffffffff)
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR", "_LO32", "_HI32", 0x0)
    self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", "_LO32", "_HI32", 0x0)

    for eng_i in range(18): self.adev.wreg_pair(f"reg{ip}VM_INVALIDATE_ENG{eng_i}_ADDR_RANGE", "_LO32", "_HI32", 0x1fffffffff)
    self.hub_initted[ip] = True

  @functools.cache
  def get_pte_flags(self, pte_lv, is_table, frag, uncached, system, snooped, valid, extra=0):
    extra |= (am.AMDGPU_PTE_SYSTEM * system) | (am.AMDGPU_PTE_SNOOPED * snooped) | (am.AMDGPU_PTE_VALID * valid) | am.AMDGPU_PTE_FRAG(frag)
    if not is_table: extra |= (am.AMDGPU_PTE_WRITEABLE | am.AMDGPU_PTE_READABLE | am.AMDGPU_PTE_EXECUTABLE)
    if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0):
      extra |= am.AMDGPU_PTE_MTYPE_GFX12(0, self.adev.soc.module.MTYPE_UC if uncached else 0)
      extra |= (am.AMDGPU_PDE_PTE_GFX12 if not is_table and pte_lv != am.AMDGPU_VM_PTB else (am.AMDGPU_PTE_IS_PTE if not is_table else 0))
    else:
      extra |= am.AMDGPU_PTE_MTYPE_NV10(0, self.adev.soc.module.MTYPE_UC if uncached else 0)
      extra |= (am.AMDGPU_PDE_PTE if not is_table and pte_lv != am.AMDGPU_VM_PTB else 0)
    return extra
  def is_pte_huge_page(self, pte): return pte & (am.AMDGPU_PDE_PTE_GFX12 if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else am.AMDGPU_PDE_PTE)

  def on_interrupt(self):
    for ip in ["MM", "GC"]:
      st = self.adev.reg(f"reg{ip}VM_L2_PROTECTION_FAULT_STATUS{'_LO32' if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else ''}").read()
      va = (self.adev.reg(f'reg{ip}VM_L2_PROTECTION_FAULT_ADDR_LO32').read()
            | (self.adev.reg(f'reg{ip}VM_L2_PROTECTION_FAULT_ADDR_HI32').read()) << 32) << 12
      if st: raise RuntimeError(f"{ip}VM_L2_PROTECTION_FAULT_STATUS: {st:#x} {va:#x}")

class AM_SMU(AM_IP):
  def init_sw(self):
    self.smu_mod = self.adev._ip_module("smu", am.MP1_HWIP, prever_prefix='v')
    self.driver_table_paddr = self.adev.mm.palloc(0x4000, zero=not self.adev.partial_boot, boot=True)

  def init_hw(self):
    self._send_msg(self.smu_mod.PPSMC_MSG_SetDriverDramAddrHigh, hi32(self.adev.paddr2mc(self.driver_table_paddr)))
    self._send_msg(self.smu_mod.PPSMC_MSG_SetDriverDramAddrLow, lo32(self.adev.paddr2mc(self.driver_table_paddr)))
    self._send_msg(self.smu_mod.PPSMC_MSG_EnableAllSmuFeatures, 0)

  def is_smu_alive(self):
    with contextlib.suppress(RuntimeError): self._send_msg(self.smu_mod.PPSMC_MSG_GetSmuVersion, 0, timeout=100)
    return self.adev.mmMP1_SMN_C2PMSG_90.read() != 0

  def mode1_reset(self):
    if DEBUG >= 2: print(f"am {self.adev.devfmt}: mode1 reset")
    if self.adev.ip_ver[am.MP0_HWIP] >= (14,0,0): self._send_msg(__DEBUGSMC_MSG_Mode1Reset:=2, 0, debug=True)
    else: self._send_msg(self.smu_mod.PPSMC_MSG_Mode1Reset, 0)
    time.sleep(0.5) # 500ms

  def read_table(self, table_t, cmd):
    self._send_msg(self.smu_mod.PPSMC_MSG_TransferTableSmu2Dram, cmd)
    return table_t.from_buffer(bytearray(self.adev.vram.view(self.driver_table_paddr, ctypes.sizeof(table_t))[:]))
  def read_metrics(self): return self.read_table(self.smu_mod.SmuMetricsExternal_t, self.smu_mod.TABLE_SMU_METRICS)

  def set_clocks(self, level):
    if not hasattr(self, 'clcks'):
      self.clcks = {}
      for clck in [self.smu_mod.PPCLK_GFXCLK, self.smu_mod.PPCLK_UCLK, self.smu_mod.PPCLK_FCLK, self.smu_mod.PPCLK_SOCCLK]:
        cnt = self._send_msg(self.smu_mod.PPSMC_MSG_GetDpmFreqByIndex, (clck<<16)|0xff, read_back_arg=True)&0x7fffffff
        self.clcks[clck] = [self._send_msg(self.smu_mod.PPSMC_MSG_GetDpmFreqByIndex, (clck<<16)|i, read_back_arg=True)&0x7fffffff for i in range(cnt)]

    for clck, vals in self.clcks.items():
      self._send_msg(self.smu_mod.PPSMC_MSG_SetSoftMinByFreq, clck << 16 | (vals[level]))
      self._send_msg(self.smu_mod.PPSMC_MSG_SetSoftMaxByFreq, clck << 16 | (vals[level]))

  def _smu_cmn_send_msg(self, msg, param=0, debug=False):
    (self.adev.mmMP1_SMN_C2PMSG_90 if not debug else self.adev.mmMP1_SMN_C2PMSG_54).write(0) # resp reg
    (self.adev.mmMP1_SMN_C2PMSG_82 if not debug else self.adev.mmMP1_SMN_C2PMSG_53).write(param)
    (self.adev.mmMP1_SMN_C2PMSG_66 if not debug else self.adev.mmMP1_SMN_C2PMSG_75).write(msg)

  def _send_msg(self, msg, param, read_back_arg=False, timeout=10000, debug=False): # 10s
    self._smu_cmn_send_msg(msg, param, debug=debug)
    self.adev.wait_reg(self.adev.mmMP1_SMN_C2PMSG_90 if not debug else self.adev.mmMP1_SMN_C2PMSG_54, mask=0xFFFFFFFF, value=1, timeout=timeout)
    return (self.adev.mmMP1_SMN_C2PMSG_82 if not debug else self.adev.mmMP1_SMN_C2PMSG_53).read() if read_back_arg else None

class AM_GFX(AM_IP):
  def init_hw(self):
    # Wait for RLC autoload to complete
    while self.adev.regCP_STAT.read() != 0 and self.adev.regRLC_RLCS_BOOTLOAD_STATUS.read_bitfields()['bootload_complete'] != 0: pass

    self._config_gfx_rs64()
    self.adev.gmc.init_hub("GC")

    # NOTE: Golden reg for gfx11. No values for this reg provided. The kernel just ors 0x20000000 to this reg.
    self.adev.regTCP_CNTL.write(self.adev.regTCP_CNTL.read() | 0x20000000)

    self.adev.regRLC_SRM_CNTL.update(srm_enable=1, auto_incr_addr=1)

    self.adev.soc.doorbell_enable(port=0, awid=0x3, awaddr_31_28_value=0x3)
    self.adev.soc.doorbell_enable(port=3, awid=0x6, awaddr_31_28_value=0x3)

    self.adev.regGRBM_CNTL.update(read_timeout=0xff)
    for i in range(0, 16):
      self._grbm_select(vmid=i)
      self.adev.regSH_MEM_CONFIG.write(address_mode=self.adev.soc.module.SH_MEM_ADDRESS_MODE_64,
                                       alignment_mode=self.adev.soc.module.SH_MEM_ALIGNMENT_MODE_UNALIGNED, initial_inst_prefetch=3)

      # Configure apertures:
      # LDS:         0x10000000'00000000 - 0x10000001'00000000 (4GB)
      # Scratch:     0x20000000'00000000 - 0x20000001'00000000 (4GB)
      self.adev.regSH_MEM_BASES.write(shared_base=0x1, private_base=0x2)
    self._grbm_select()

    # Configure MEC doorbell range
    self.adev.regCP_MEC_DOORBELL_RANGE_LOWER.write(0x0)
    self.adev.regCP_MEC_DOORBELL_RANGE_UPPER.write(0x450)

    # Enable MEC
    self.adev.regCP_MEC_RS64_CNTL.update(mec_invalidate_icache=0, mec_pipe0_reset=0, mec_pipe0_active=1, mec_halt=0)

    # NOTE: Wait for MEC to be ready. The kernel does udelay here as well.
    time.sleep(0.05)

  def fini_hw(self):
    self._grbm_select(me=1, pipe=0, queue=0)
    self.adev.regCP_HQD_DEQUEUE_REQUEST.write(0x2) # 1 - DRAIN_PIPE; 2 - RESET_WAVES
    self.adev.regSPI_COMPUTE_QUEUE_RESET.write(1)
    self._grbm_select()
    self.adev.regGCVM_CONTEXT0_CNTL.write(0)

  def setup_ring(self, ring_addr:int, ring_size:int, rptr_addr:int, wptr_addr:int, eop_addr:int, eop_size:int, doorbell:int, pipe:int, queue:int):
    mqd = self.adev.mm.valloc(0x1000, uncached=True, contigous=True)

    struct_t = getattr(am, f"struct_v{self.adev.ip_ver[am.GC_HWIP][0]}_compute_mqd")
    mqd_struct = struct_t(header=0xC0310800, cp_mqd_base_addr_lo=lo32(mqd.va_addr), cp_mqd_base_addr_hi=hi32(mqd.va_addr),
      cp_hqd_persistent_state=self.adev.regCP_HQD_PERSISTENT_STATE.encode(preload_size=0x55, preload_req=1),
      cp_hqd_pipe_priority=0x2, cp_hqd_queue_priority=0xf, cp_hqd_quantum=0x111,
      cp_hqd_pq_base_lo=lo32(ring_addr>>8), cp_hqd_pq_base_hi=hi32(ring_addr>>8),
      cp_hqd_pq_rptr_report_addr_lo=lo32(rptr_addr), cp_hqd_pq_rptr_report_addr_hi=hi32(rptr_addr),
      cp_hqd_pq_wptr_poll_addr_lo=lo32(wptr_addr), cp_hqd_pq_wptr_poll_addr_hi=hi32(wptr_addr),
      cp_hqd_pq_doorbell_control=self.adev.regCP_HQD_PQ_DOORBELL_CONTROL.encode(doorbell_offset=doorbell*2, doorbell_en=1),
      cp_hqd_pq_control=self.adev.regCP_HQD_PQ_CONTROL.encode(rptr_block_size=5, unord_dispatch=0, queue_size=(ring_size//4).bit_length()-2),
      cp_hqd_ib_control=self.adev.regCP_HQD_IB_CONTROL.encode(min_ib_avail_size=0x3), cp_hqd_hq_status0=0x20004000,
      cp_mqd_control=self.adev.regCP_MQD_CONTROL.encode(priv_state=1), cp_hqd_vmid=0,
      cp_hqd_eop_base_addr_lo=lo32(eop_addr>>8), cp_hqd_eop_base_addr_hi=hi32(eop_addr>>8),
      cp_hqd_eop_control=self.adev.regCP_HQD_EOP_CONTROL.encode(eop_size=(eop_size//4).bit_length()-2))

    # Copy mqd into memory
    self.adev.vram.view(mqd.paddrs[0][0], ctypes.sizeof(mqd_struct))[:] = memoryview(mqd_struct).cast('B')
    self.adev.gmc.flush_hdp()

    self._grbm_select(me=1, pipe=pipe, queue=queue)

    mqd_st_mv = to_mv(ctypes.addressof(mqd_struct), ctypes.sizeof(mqd_struct)).cast('I')
    for i, reg in enumerate(range(self.adev.regCP_MQD_BASE_ADDR.addr, self.adev.regCP_HQD_PQ_WPTR_HI.addr + 1)):
      self.adev.wreg(reg, mqd_st_mv[0x80 + i])
    self.adev.regCP_HQD_ACTIVE.write(0x1)

    self._grbm_select()

    self.adev.reg(f"regCP_ME1_PIPE{pipe}_INT_CNTL").update(time_stamp_int_enable=1, generic0_int_enable=1)

  def set_clockgating_state(self):
    if hasattr(self.adev, 'regMM_ATC_L2_MISC_CG'): self.adev.regMM_ATC_L2_MISC_CG.write(enable=1, mem_ls_enable=1)

    self.adev.regRLC_SAFE_MODE.write(message=1, cmd=1)
    self.adev.wait_reg(self.adev.regRLC_SAFE_MODE, mask=0x1, value=0x0)

    self.adev.regRLC_CGCG_CGLS_CTRL.update(cgcg_gfx_idle_threshold=0x36, cgcg_en=1, cgls_rep_compansat_delay=0xf, cgls_en=1)

    self.adev.regCP_RB_WPTR_POLL_CNTL.update(poll_frequency=0x100, idle_poll_count=0x90)
    self.adev.regCP_INT_CNTL.update(cntx_busy_int_enable=1, cntx_empty_int_enable=1, cmp_busy_int_enable=1, gfx_idle_int_enable=1)
    self.adev.regSDMA0_RLC_CGCG_CTRL.update(cgcg_int_enable=1)
    self.adev.regSDMA1_RLC_CGCG_CTRL.update(cgcg_int_enable=1)

    self.adev.regRLC_CGTT_MGCG_OVERRIDE.update(perfmon_clock_state=0, gfxip_fgcg_override=0, gfxip_repeater_fgcg_override=0,
      grbm_cgtt_sclk_override=0, rlc_cgtt_sclk_override=0, gfxip_mgcg_override=0, gfxip_cgls_override=0, gfxip_cgcg_override=0)

    self.adev.regRLC_SAFE_MODE.write(message=0, cmd=1)

  def _grbm_select(self, me=0, pipe=0, queue=0, vmid=0): self.adev.regGRBM_GFX_CNTL.write(meid=me, pipeid=pipe, vmid=vmid, queueid=queue)

  def _config_gfx_rs64(self):
    def _config_helper(eng_name, cntl_reg, eng_reg, pipe_cnt, me=0):
      for pipe in range(pipe_cnt):
        self._grbm_select(me=me, pipe=pipe)
        self.adev.wreg_pair(f"regCP_{eng_reg}_PRGRM_CNTR_START", "", "_HI", self.adev.fw.ucode_start[eng_name] >> 2)
      self._grbm_select()
      self.adev.reg(f"regCP_{cntl_reg}_CNTL").update(**{f"{eng_name.lower()}_pipe{pipe}_reset": 1 for pipe in range(pipe_cnt)})
      self.adev.reg(f"regCP_{cntl_reg}_CNTL").update(**{f"{eng_name.lower()}_pipe{pipe}_reset": 0 for pipe in range(pipe_cnt)})

    if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0):
      _config_helper(eng_name="PFP", cntl_reg="ME", eng_reg="PFP", pipe_cnt=1)
      _config_helper(eng_name="ME", cntl_reg="ME", eng_reg="ME", pipe_cnt=1)
    _config_helper(eng_name="MEC", cntl_reg="MEC_RS64", eng_reg="MEC_RS64", pipe_cnt=1, me=1)

class AM_IH(AM_IP):
  def init_sw(self):
    self.ring_size = 512 << 10
    def _alloc_ring(size): return (self.adev.mm.palloc(size, zero=False, boot=True), self.adev.mm.palloc(0x1000, zero=False, boot=True))
    self.rings = [(*_alloc_ring(self.ring_size), "", 0), (*_alloc_ring(self.ring_size), "_RING1", 1)]

  def init_hw(self):
    for ring_vm, rwptr_vm, suf, ring_id in self.rings:
      self.adev.wreg_pair("regIH_RB_BASE", suf, f"_HI{suf}", self.adev.paddr2mc(ring_vm) >> 8)

      self.adev.reg(f"regIH_RB_CNTL{suf}").write(mc_space=4, wptr_overflow_clear=1, rb_size=(self.ring_size//4).bit_length(),
        mc_snoop=1, mc_ro=0, mc_vmid=0, **({'wptr_overflow_enable': 1, 'rptr_rearm': 1} if ring_id == 0 else {'rb_full_drain_enable': 1}))

      if ring_id == 0: self.adev.wreg_pair("regIH_RB_WPTR_ADDR", "_LO", "_HI", self.adev.paddr2mc(rwptr_vm))

      self.adev.reg(f"regIH_RB_WPTR{suf}").write(0)
      self.adev.reg(f"regIH_RB_RPTR{suf}").write(0)

      self.adev.reg(f"regIH_DOORBELL_RPTR{suf}").write(offset=(am.AMDGPU_NAVI10_DOORBELL_IH + ring_id) * 2, enable=1)

    self.adev.regIH_STORM_CLIENT_LIST_CNTL.update(client18_is_storm_client=1)
    self.adev.regIH_INT_FLOOD_CNTL.update(flood_cntl_enable=1)
    self.adev.regIH_MSI_STORM_CTRL.update(delay=3)

    # toggle interrupts
    for _, rwptr_vm, suf, ring_id in self.rings:
      self.adev.reg(f"regIH_RB_CNTL{suf}").update(rb_enable=1, **({'enable_intr': 1} if ring_id == 0 else {}))

    self.adev.soc.doorbell_enable(port=1, awid=0x0, awaddr_31_28_value=0x0, offset=am.AMDGPU_NAVI10_DOORBELL_IH*2, size=2)

  def interrupt_handler(self):
    _, rwptr_vm, suf, _ = self.rings[0]
    wptr = self.adev.vram.view(offset=rwptr_vm, size=8, fmt='Q')[0]

    if self.adev.reg(f"regIH_RB_WPTR{suf}").read_bitfields()['rb_overflow']:
      self.adev.reg(f"regIH_RB_WPTR{suf}").update(rb_overflow=0)
      self.adev.reg(f"regIH_RB_CNTL{suf}").update(wptr_overflow_clear=1)
      self.adev.reg(f"regIH_RB_CNTL{suf}").update(wptr_overflow_clear=0)
    self.adev.regIH_RB_RPTR.write(wptr % self.ring_size)

class AM_SDMA(AM_IP):
  def init_sw(self): self.sdma_name = "F32" if self.adev.ip_ver[am.SDMA0_HWIP] < (7,0,0) else "MCU"
  def init_hw(self):
    for pipe in range(2):
      self.adev.reg(f"regSDMA{pipe}_WATCHDOG_CNTL").update(queue_hang_count=100) # 10s, 100ms per unit
      self.adev.reg(f"regSDMA{pipe}_UTCL1_CNTL").update(resp_mode=3, redo_delay=9)

      # rd=noa, wr=bypass
      self.adev.reg(f"regSDMA{pipe}_UTCL1_PAGE").update(rd_l2_policy=0x2, wr_l2_policy=0x3, **({'llc_noalloc':1} if self.sdma_name == "F32" else {}))
      self.adev.reg(f"regSDMA{pipe}_{self.sdma_name}_CNTL").update(halt=0, **{f"{'th1_' if self.sdma_name == 'F32' else ''}reset":0})
      self.adev.reg(f"regSDMA{pipe}_CNTL").update(ctxempty_int_enable=1, trap_enable=1)
    self.adev.soc.doorbell_enable(port=2, awid=0xe, awaddr_31_28_value=0x3, offset=am.AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0*2, size=4)

  def fini_hw(self):
    self.adev.regSDMA0_QUEUE0_RB_CNTL.update(rb_enable=0)
    self.adev.regSDMA0_QUEUE0_IB_CNTL.update(ib_enable=0)
    self.adev.regGRBM_SOFT_RESET.write(soft_reset_sdma0=1)
    time.sleep(0.01)
    self.adev.regGRBM_SOFT_RESET.write(0x0)

  def setup_ring(self, ring_addr:int, ring_size:int, rptr_addr:int, wptr_addr:int, doorbell:int, pipe:int, queue:int):
    # Setup the ring
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_MINOR_PTR_UPDATE").write(0x1)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_RPTR", "", "_HI", 0)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_WPTR", "", "_HI", 0)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_BASE", "", "_HI", ring_addr >> 8)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_RPTR_ADDR", "_LO", "_HI", rptr_addr)
    self.adev.wreg_pair(f"regSDMA{pipe}_QUEUE{queue}_RB_WPTR_POLL_ADDR", "_LO", "_HI", wptr_addr)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_DOORBELL_OFFSET").update(offset=doorbell * 2)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_DOORBELL").update(enable=1)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_MINOR_PTR_UPDATE").write(0x0)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_RB_CNTL").write(rb_vmid=0, rptr_writeback_enable=1, rptr_writeback_timer=4,
      **{f'{self.sdma_name.lower()}_wptr_poll_enable':1}, rb_size=(ring_size//4).bit_length()-1, rb_enable=1, rb_priv=1)
    self.adev.reg(f"regSDMA{pipe}_QUEUE{queue}_IB_CNTL").update(ib_enable=1)

class AM_PSP(AM_IP):
  def init_sw(self):
    self.reg_pref = "regMP0_SMN_C2PMSG" if self.adev.ip_ver[am.MP0_HWIP] < (14,0,0) else "regMPASP_SMN_C2PMSG"
    self.msg1_paddr = self.adev.mm.palloc(am.PSP_1_MEG, align=am.PSP_1_MEG, zero=False, boot=True)
    self.cmd_paddr = self.adev.mm.palloc(am.PSP_CMD_BUFFER_SIZE, zero=False, boot=True)
    self.fence_paddr = self.adev.mm.palloc(am.PSP_FENCE_BUFFER_SIZE, zero=not self.adev.partial_boot, boot=True)

    self.ring_size = 0x10000
    self.ring_paddr = self.adev.mm.palloc(self.ring_size, zero=False, boot=True)

    self.max_tmr_size = 0x1300000
    self.boot_time_tmr = self.adev.ip_ver[am.GC_HWIP] >= (12,0,0)
    if not self.boot_time_tmr:
      self.tmr_paddr = self.adev.mm.palloc(self.max_tmr_size, align=am.PSP_TMR_ALIGNMENT, zero=False, boot=True)

  def init_hw(self):
    spl_key = am.PSP_FW_TYPE_PSP_SPL if self.adev.ip_ver[am.MP0_HWIP] >= (14,0,0) else am.PSP_FW_TYPE_PSP_KDB
    sos_components = [(am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_KEY_DATABASE), (spl_key, am.PSP_BL__LOAD_TOS_SPL_TABLE),
      (am.PSP_FW_TYPE_PSP_SYS_DRV, am.PSP_BL__LOAD_SYSDRV), (am.PSP_FW_TYPE_PSP_SOC_DRV, am.PSP_BL__LOAD_SOCDRV),
      (am.PSP_FW_TYPE_PSP_INTF_DRV, am.PSP_BL__LOAD_INTFDRV), (am.PSP_FW_TYPE_PSP_DBG_DRV, am.PSP_BL__LOAD_DBGDRV),
      (am.PSP_FW_TYPE_PSP_RAS_DRV, am.PSP_BL__LOAD_RASDRV), (am.PSP_FW_TYPE_PSP_SOS, am.PSP_BL__LOAD_SOSDRV)]

    if not self.is_sos_alive():
      for fw, compid in sos_components: self._bootloader_load_component(fw, compid)
      while not self.is_sos_alive(): time.sleep(0.01)

    self._ring_create()
    self._tmr_init()

    # SMU fw should be loaded before TMR.
    self._load_ip_fw_cmd(*self.adev.fw.smu_psp_desc)
    if not self.boot_time_tmr: self._tmr_load_cmd()

    for psp_desc in self.adev.fw.descs: self._load_ip_fw_cmd(*psp_desc)
    self._rlc_autoload_cmd()

  def is_sos_alive(self): return self.adev.reg(f"{self.reg_pref}_81").read() != 0x0

  def _wait_for_bootloader(self): self.adev.wait_reg(self.adev.reg(f"{self.reg_pref}_35"), mask=0x80000000, value=0x80000000)

  def _prep_msg1(self, data):
    self.adev.vram.view(self.msg1_paddr, len(data))[:] = data
    self.adev.vram[self.msg1_paddr + len(data)] = 0
    self.adev.gmc.flush_hdp()

  def _bootloader_load_component(self, fw, compid):
    if fw not in self.adev.fw.sos_fw: return 0

    self._wait_for_bootloader()

    if DEBUG >= 2: print(f"am {self.adev.devfmt}: loading sos component: {am.psp_fw_type__enumvalues[fw]}")

    self._prep_msg1(self.adev.fw.sos_fw[fw])
    self.adev.reg(f"{self.reg_pref}_36").write(self.adev.paddr2mc(self.msg1_paddr) >> 20)
    self.adev.reg(f"{self.reg_pref}_35").write(compid)

    return self._wait_for_bootloader() if compid != am.PSP_BL__LOAD_SOSDRV else 0

  def _tmr_init(self):
    # Load TOC and calculate TMR size
    self._prep_msg1(fwm:=self.adev.fw.sos_fw[am.PSP_FW_TYPE_PSP_TOC])
    self.tmr_size = self._load_toc_cmd(len(fwm)).resp.tmr_size
    assert self.tmr_size <= self.max_tmr_size

  def _ring_create(self):
    # If the ring is already created, destroy it
    if self.adev.reg(f"{self.reg_pref}_71").read() != 0:
      self.adev.reg(f"{self.reg_pref}_64").write(am.GFX_CTRL_CMD_ID_DESTROY_RINGS)

      # There might be handshake issue with hardware which needs delay
      time.sleep(0.02)

    # Wait until the sOS is ready
    self.adev.wait_reg(self.adev.reg(f"{self.reg_pref}_64"), mask=0x80000000, value=0x80000000)

    self.adev.wreg_pair(self.reg_pref, "_69", "_70", self.adev.paddr2mc(self.ring_paddr))
    self.adev.reg(f"{self.reg_pref}_71").write(self.ring_size)
    self.adev.reg(f"{self.reg_pref}_64").write(am.PSP_RING_TYPE__KM << 16)

    # There might be handshake issue with hardware which needs delay
    time.sleep(0.02)

    self.adev.wait_reg(self.adev.reg(f"{self.reg_pref}_64"), mask=0x8000FFFF, value=0x80000000)

  def _ring_submit(self, cmd):
    msg = am.struct_psp_gfx_rb_frame(fence_value=(prev_wptr:=self.adev.reg(f"{self.reg_pref}_67").read()),
      cmd_buf_addr_lo=lo32(self.adev.paddr2mc(self.cmd_paddr)), cmd_buf_addr_hi=hi32(self.adev.paddr2mc(self.cmd_paddr)),
      fence_addr_lo=lo32(self.adev.paddr2mc(self.fence_paddr)), fence_addr_hi=hi32(self.adev.paddr2mc(self.fence_paddr)))

    self.adev.vram.view(self.cmd_paddr, ctypes.sizeof(cmd))[:] = memoryview(cmd).cast('B')
    self.adev.vram.view(self.ring_paddr + prev_wptr * 4, ctypes.sizeof(msg))[:] = memoryview(msg).cast('B')

    # Move the wptr
    self.adev.reg(f"{self.reg_pref}_67").write(prev_wptr + ctypes.sizeof(am.struct_psp_gfx_rb_frame) // 4)

    while self.adev.vram.view(self.fence_paddr, 4, 'I')[0] != prev_wptr: pass
    time.sleep(0.005)

    resp = type(cmd).from_buffer(bytearray(self.adev.vram.view(self.cmd_paddr, ctypes.sizeof(cmd))[:]))
    if resp.resp.status != 0: raise RuntimeError(f"PSP command failed {resp.cmd_id} {resp.resp.status}")

    return resp

  def _load_ip_fw_cmd(self, fw_types, fw_bytes):
    self._prep_msg1(fw_bytes)
    for fw_type in fw_types:
      if DEBUG >= 2: print(f"am {self.adev.devfmt}: loading fw: {am.psp_gfx_fw_type__enumvalues[fw_type]}")
      cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_LOAD_IP_FW)
      cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi, cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = data64(self.adev.paddr2mc(self.msg1_paddr))
      cmd.cmd.cmd_load_ip_fw.fw_size = len(fw_bytes)
      cmd.cmd.cmd_load_ip_fw.fw_type = fw_type
      self._ring_submit(cmd)

  def _tmr_load_cmd(self):
    cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_SETUP_TMR)
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_hi, cmd.cmd.cmd_setup_tmr.buf_phy_addr_lo = data64(self.adev.paddr2mc(self.tmr_paddr))
    cmd.cmd.cmd_setup_tmr.system_phy_addr_hi, cmd.cmd.cmd_setup_tmr.system_phy_addr_lo = data64(self.tmr_paddr)
    cmd.cmd.cmd_setup_tmr.bitfield.virt_phy_addr = 1
    cmd.cmd.cmd_setup_tmr.buf_size = self.tmr_size
    return self._ring_submit(cmd)

  def _load_toc_cmd(self, toc_size):
    cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_LOAD_TOC)
    cmd.cmd.cmd_load_toc.toc_phy_addr_hi, cmd.cmd.cmd_load_toc.toc_phy_addr_lo = data64(self.adev.paddr2mc(self.msg1_paddr))
    cmd.cmd.cmd_load_toc.toc_size = toc_size
    return self._ring_submit(cmd)

  def _rlc_autoload_cmd(self): return self._ring_submit(am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_AUTOLOAD_RLC))
