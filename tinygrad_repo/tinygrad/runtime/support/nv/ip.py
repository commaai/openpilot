from __future__ import annotations
import ctypes, time, array, struct, itertools, dataclasses
from typing import cast, Any
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu, pci
from tinygrad.helpers import lo32, hi32, DEBUG, round_up, round_down, fetch_fw, wait_cond, ceildiv
from tinygrad.runtime.support.system import System, MMIOInterface
from tinygrad.runtime.support.elf import elf_loader

@dataclasses.dataclass(frozen=True)
class GRBufDesc: size:int; virt:bool; phys:bool; local:bool=False # noqa: E702

class NV_IP:
  def __init__(self, nvdev): self.nvdev = nvdev
  def init_sw(self): pass # Prepare sw/allocations for this IP
  def init_hw(self): pass # Initialize hw for this IP
  def fini_hw(self): pass # Finalize hw for this IP

class NVRpcQueue:
  def __init__(self, gsp:NV_GSP, view:MMIOInterface, completion_q_view:MMIOInterface|None=None):
    self.tx_view = view.view(fmt='I')
    wait_cond(lambda: self.tx_view[getattr(nv.msgqTxHeader, 'entryOff').offset // 4], value=0x1000, msg="RPC queue not initialized")
    self.tx = nv.msgqTxHeader.from_buffer_copy(bytes(view[:ctypes.sizeof(nv.msgqTxHeader)]))

    if completion_q_view is not None:
      comp_tx = nv.msgqTxHeader.from_buffer_copy(bytes(completion_q_view[:ctypes.sizeof(nv.msgqTxHeader)]))
      self.rx_view = completion_q_view.view(comp_tx.rxHdrOff, fmt='I')

    self.gsp, self.view, self.seq = gsp, view, 0
    self.queue_mv = view.view(self.tx.entryOff, self.tx.msgSize * self.tx.msgCount)

  def _checksum(self, data:bytes):
    if (pad_len:=(-len(data)) % 8): data += b'\x00' * pad_len
    checksum = 0
    for offset in range(0, len(data), 8): checksum ^= struct.unpack_from('Q', data, offset)[0]
    return hi32(checksum) ^ lo32(checksum)

  def _send_rpc_record(self, func:int, msg:bytes):
    header = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, rpc_result=nv.NV_VGPU_MSG_RESULT_RPC_PENDING,
      rpc_result_private=nv.NV_VGPU_MSG_RESULT_RPC_PENDING, header_version=(3<<24), function=func, length=len(msg) + 0x20)

    msg = bytes(header) + msg
    phdr = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=ceildiv(len(msg) + ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT), self.tx.msgSize), seqNum=self.seq)
    phdr.checkSum = self._checksum(bytes(phdr) + msg)
    msg = (bytes(phdr) + msg).ljust(phdr.elemCount * self.tx.msgSize, b'\x00')

    wp = self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4]
    off, first = wp * self.tx.msgSize, min(len(msg), len(self.queue_mv) - wp * self.tx.msgSize)
    self.queue_mv[off:off+first] = msg[:first]
    if first < len(msg): self.queue_mv[:len(msg)-first] = msg[first:]
    self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4] = (wp + phdr.elemCount) % self.tx.msgCount
    System.memory_barrier()

    self.seq += 1
    self.gsp.nvdev.NV_PGSP_QUEUE_HEAD[0].write(0x0)

  def send_rpc(self, func:int, msg:bytes):
    max_payload = self.tx.msgSize * 16 - ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT) - ctypes.sizeof(nv.rpc_message_header_v)
    self._send_rpc_record(func, msg[:max_payload])
    for off in range(max_payload, len(msg), max_payload): self._send_rpc_record(nv.NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD, msg[off:off+max_payload])

  def read_resp(self):
    System.memory_barrier()
    while self.rx_view[0] != self.tx_view[getattr(nv.msgqTxHeader, 'writePtr').offset // 4]:
      off = self.rx_view[0] * self.tx.msgSize
      hdr = nv.rpc_message_header_v.from_buffer_copy(bytes(self.queue_mv[off + 0x30 : off + 0x30 + ctypes.sizeof(nv.rpc_message_header_v)]))
      msg = bytes(self.queue_mv[off + 0x50 : off + 0x50 + hdr.length])

      # Handling special functions
      if hdr.function == nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER: self.gsp.run_cpu_seq(msg)
      elif hdr.function == nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG:
        print(f"nv {self.gsp.nvdev.devfmt}: GSP LOG: {msg[12:].rstrip(bytes([0])).decode('utf-8')}")

      self.gsp.nvdev.is_err_state |= hdr.function in {nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG, nv.NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED}

      # Update the read pointer
      self.rx_view[0] = (self.rx_view[0] + round_up(hdr.length, self.tx.msgSize) // self.tx.msgSize) % self.tx.msgCount
      System.memory_barrier()

      if DEBUG >= 3:
        nm = nv.rpc_fns.get(hdr.function, nv.rpc_events.get(hdr.function, f'ev:{hdr.function:x}'))
        print(f"nv {self.gsp.nvdev.devfmt}: in RPC: {nm}, res:{hdr.rpc_result:#x}")

      if hdr.rpc_result != 0: raise RuntimeError(f"RPC call {hdr.function} failed with result {hdr.rpc_result}")
      yield hdr.function, msg

  def wait_resp(self, cmd:int, timeout:int=10000) -> bytes:
    start_time = int(time.perf_counter() * 1000)
    while (int(time.perf_counter() * 1000) - start_time) < timeout:
      if (msg:=next((message for func, message in self.read_resp() if func == cmd), None)) is not None: return msg
    raise RuntimeError(f"Timeout waiting for RPC response for command {cmd}")

class NV_FLCN(NV_IP):
  def wait_for_reset(self):
    wait_cond(lambda _: self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.read_bitfields()['read_protection_level0'] == 1 and
                        self.nvdev.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05[0].read() & 0xff == 0xff, "waiting for reset")

  def init_sw(self):
    self.nvdev.include("dev_gsp", "ga102")
    self.nvdev.include("dev_falcon_v4", "ga102")
    self.nvdev.include("dev_riscv_pri", "ga102")
    self.nvdev.include("dev_fbif_v4", "ga102")
    self.nvdev.include("dev_falcon_second_pri", "ga102")
    self.nvdev.include("dev_sec_pri", "ga102")
    self.nvdev.include("dev_bus", "tu102")

    self.prep_ucode()
    self.prep_booter()

  def prep_ucode(self):
    vbios_bytes, vbios_off = memoryview(bytes(array.array('I', self.nvdev.mmio[0x00300000//4:(0x00300000+0x100000)//4]))), 0
    while True:
      pci_blck = vbios_bytes[vbios_off + nv.OFFSETOF_PCI_EXP_ROM_PCI_DATA_STRUCT_PTR:].cast('H')[0]
      imglen = vbios_bytes[vbios_off + pci_blck + nv.OFFSETOF_PCI_DATA_STRUCT_IMAGE_LEN:].cast('H')[0] * nv.PCI_ROM_IMAGE_BLOCK_SIZE
      match vbios_bytes[vbios_off + pci_blck + nv.OFFSETOF_PCI_DATA_STRUCT_CODE_TYPE]:
        case nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_BASE: block_size = imglen
        case nv.NV_BCRT_HASH_INFO_BASE_CODE_TYPE_VBIOS_EXT:
          expansion_rom_off = vbios_off - block_size
          break
      vbios_off += imglen

    bit_header = nv.BIT_HEADER_V1_00.from_buffer_copy(vbios_bytes[(bit_addr:=0x1b0):bit_addr + ctypes.sizeof(nv.BIT_HEADER_V1_00)])
    assert bit_header.Signature == 0x00544942, f"Invalid BIT header signature {hex(bit_header.Signature)}"

    for i in range(bit_header.TokenEntries):
      bit = nv.BIT_TOKEN_V1_00.from_buffer_copy(vbios_bytes[bit_addr + bit_header.HeaderSize + i * bit_header.TokenSize:])
      if bit.TokenId != nv.BIT_TOKEN_FALCON_DATA or bit.DataVersion != 2 or bit.DataSize < nv.BIT_DATA_FALCON_DATA_V2_SIZE_4: continue

      falcon_data = nv.BIT_DATA_FALCON_DATA_V2.from_buffer_copy(vbios_bytes[bit.DataPtr & 0xffff:])
      ucode_hdr = nv.FALCON_UCODE_TABLE_HDR_V1.from_buffer_copy(vbios_bytes[(table_ptr:=expansion_rom_off + falcon_data.FalconUcodeTablePtr):])
      for j in range(ucode_hdr.EntryCount):
        ucode_entry = nv.FALCON_UCODE_TABLE_ENTRY_V1.from_buffer_copy(vbios_bytes[table_ptr + ucode_hdr.HeaderSize + j * ucode_hdr.EntrySize:])
        if ucode_entry.ApplicationID != nv.FALCON_UCODE_ENTRY_APPID_FWSEC_PROD: continue

        ucode_desc_hdr = nv.FALCON_UCODE_DESC_HEADER.from_buffer_copy(vbios_bytes[expansion_rom_off + ucode_entry.DescPtr:])
        ucode_desc_off = expansion_rom_off + ucode_entry.DescPtr
        ucode_desc_size = ucode_desc_hdr.vDesc >> 16

    self.desc_v3 = nv.FALCON_UCODE_DESC_V3.from_buffer_copy(vbios_bytes[ucode_desc_off:ucode_desc_off + ucode_desc_size])

    sig_total_size = ucode_desc_size - nv.FALCON_UCODE_DESC_V3_SIZE_44
    signature = vbios_bytes[ucode_desc_off + nv.FALCON_UCODE_DESC_V3_SIZE_44:][:sig_total_size]
    image = vbios_bytes[ucode_desc_off + ucode_desc_size:][:round_up(self.desc_v3.StoredSize, 256)]

    self.frts_offset = self.nvdev.vram_size - 0x100000 - 0x100000
    read_vbios_desc = nv.FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_READ_VBIOS_DESC), flags=2)
    frst_reg_desc = nv.FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(nv.FWSECLIC_FRTS_REGION_DESC),
      frtsRegionOffset4K=self.frts_offset >> 12, frtsRegionSize=0x100, frtsRegionMediaType=2)
    frts_cmd = nv.FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc)

    def __patch(cmd_id, cmd):
      patched_image = bytearray(image)

      dmem_offset = 0
      hdr = nv.FALCON_APPLICATION_INTERFACE_HEADER_V1.from_buffer_copy(image[(app_hdr_off:=self.desc_v3.IMEMLoadSize+self.desc_v3.InterfaceOffset):])
      ents = (nv.FALCON_APPLICATION_INTERFACE_ENTRY_V1 * hdr.entryCount).from_buffer_copy(image[app_hdr_off + ctypes.sizeof(hdr):])
      for i in range(hdr.entryCount):
        if ents[i].id == nv.FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER: dmem_offset = ents[i].dmemOffset

      # Patch image
      dmem = nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[(dmem_mapper_offset:=self.desc_v3.IMEMLoadSize+dmem_offset):])
      dmem.init_cmd = cmd_id
      patched_image[dmem_mapper_offset : dmem_mapper_offset+len(bytes(dmem))] = bytes(dmem)
      patched_image[(cmd_off:=self.desc_v3.IMEMLoadSize+dmem.cmd_in_buffer_offset) : cmd_off+len(cmd)] = cmd
      patched_image[(sig_off:=self.desc_v3.IMEMLoadSize+self.desc_v3.PKCDataOffset) : sig_off+0x180] = signature[-0x180:]

      return self.nvdev._alloc_boot_mem(len(patched_image), data=patched_image, sysmem=False)

    _, self.frts_image_paddr, _ = __patch(0x15, bytes(frts_cmd))

  def prep_booter(self):
    sha = {"ga102":"4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b",
           "ad102":"8b293e19b637c5e22c87a2428d1c71bb13e0904e8a88ac6b3c6c1f2679c6e37a"}[self.nvdev.fw_name]
    h = nv.struct_nvfw_bin_hdr.from_buffer_copy(b:=fetch_fw(f"nvidia/{self.nvdev.fw_name}/gsp", "booter_load-570.144.bin", sha))
    lh = nv.struct_nvfw_hs_load_header_v2.from_buffer_copy(b, (hs:=nv.struct_nvfw_hs_header_v2.from_buffer_copy(b, h.header_offset)).header_offset)
    app = nv.struct_nvfw_hs_load_header_v2_app.from_buffer_copy(b, hs.header_offset + ctypes.sizeof(nv.struct_nvfw_hs_load_header_v2))

    patch_loc, patch_sig = struct.unpack_from("<I", b, hs.patch_loc)[0], struct.unpack_from("<I", b, hs.patch_sig)[0]
    sig = b[(sig_off:=hs.sig_prod_offset + patch_sig):sig_off + (sig_len:=hs.sig_prod_size // struct.unpack_from("<I", b, hs.num_sig)[0])]

    (patched_image:=bytearray(b[h.data_offset:h.data_offset + h.data_size]))[patch_loc:patch_loc+sig_len] = sig

    _, self.booter_image_paddr, _ = self.nvdev._alloc_boot_mem(len(patched_image), data=patched_image, sysmem=False)
    self.booter_data_off, self.booter_data_sz, self.booter_code_off, self.booter_code_sz = lh.os_data_offset, lh.os_data_size, app.offset, app.size

  def init_hw(self):
    self.falcon, self.sec2 = 0x00110000, 0x00840000

    self.reset(self.falcon)
    self.execute_hs(self.falcon, self.frts_image_paddr, code_off=0x0, data_off=self.desc_v3.IMEMLoadSize,
      imemPa=self.desc_v3.IMEMPhysBase, imemVa=self.desc_v3.IMEMVirtBase, imemSz=self.desc_v3.IMEMLoadSize,
      dmemPa=self.desc_v3.DMEMPhysBase, dmemVa=0x0, dmemSz=self.desc_v3.DMEMLoadSize,
      pkc_off=self.desc_v3.PKCDataOffset, engid=self.desc_v3.EngineIdMask, ucodeid=self.desc_v3.UcodeId)
    assert self.nvdev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read() != 0, "WPR2 is not initialized"

    self.reset(self.falcon, riscv=True)

    # set up the mailbox
    self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.nvdev.gsp.libos_args_sysmem))
    self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.nvdev.gsp.libos_args_sysmem))

    # booter
    self.reset(self.sec2)
    mbx = self.execute_hs(self.sec2, self.booter_image_paddr, code_off=self.booter_code_off, data_off=self.booter_data_off,
      imemPa=0x0, imemVa=self.booter_code_off, imemSz=self.booter_code_sz, dmemPa=0x0, dmemVa=0x0, dmemSz=self.booter_data_sz,
      pkc_off=0x10, engid=1, ucodeid=3, mailbox=self.nvdev.gsp.wpr_meta_sysmem)
    assert mbx[0] == 0x0, f"Booter failed to execute, mailbox is {mbx[0]:08x}, {mbx[1]:08x}"

    self.nvdev.NV_PFALCON_FALCON_OS.with_base(self.falcon).write(0x0)
    assert self.nvdev.NV_PRISCV_RISCV_CPUCTL.with_base(self.falcon).read_bitfields()['active_stat'] == 1, "GSP Core is not active"

  def execute_dma(self, base:int, cmd:int, dest:int, mem_off:int, src:int, size:int):
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'], value=0, msg="DMA does not progress")

    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE.with_base(base).write(lo32(src >> 8))
    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE1.with_base(base).write(hi32(src >> 8) & 0x1ff)

    xfered = 0
    while xfered < size:
      wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'], value=0, msg="DMA does not progress")

      self.nvdev.NV_PFALCON_FALCON_DMATRFMOFFS.with_base(base).write(dest + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFFBOFFS.with_base(base).write(mem_off + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).write(cmd)
      xfered += 256

    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['idle'], msg="DMA does not complete")

  def start_cpu(self, base:int):
    if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['alias_en'] == 1:
      self.nvdev.wreg(base + self.nvdev.NV_PFALCON_FALCON_CPUCTL_ALIAS, 0x2)
    else: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).write(startcpu=1)

  def wait_cpu_halted(self, base): wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['halted'], msg="not halted")

  def execute_hs(self, base, img_paddr, code_off, data_off, imemPa, imemVa, imemSz, dmemPa, dmemVa, dmemSz, pkc_off, engid, ucodeid, mailbox=None):
    self.disable_ctx_req(base)

    # target=0 is FB (not in published headers)
    self.nvdev.NV_PFALCON_FBIF_TRANSCFG.with_base(base)[ctx_dma:=0].update(target=0, mem_type=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=1, sec=1)
    self.execute_dma(base, cmd, dest=imemPa, mem_off=imemVa, src=img_paddr+code_off-imemVa, size=imemSz)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=0, sec=0)
    self.execute_dma(base, cmd, dest=dmemPa, mem_off=dmemVa, src=img_paddr+data_off-dmemVa, size=dmemSz)

    self.nvdev.NV_PFALCON2_FALCON_BROM_PARAADDR.with_base(base)[0].write(pkc_off)
    self.nvdev.NV_PFALCON2_FALCON_BROM_ENGIDMASK.with_base(base).write(engid)
    self.nvdev.NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID.with_base(base).write(val=ucodeid)
    self.nvdev.NV_PFALCON2_FALCON_MOD_SEL.with_base(base).write(algo=self.nvdev.NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K)

    self.nvdev.NV_PFALCON_FALCON_BOOTVEC.with_base(base).write(imemVa)

    if mailbox is not None:
      self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).write(lo32(mailbox))
      self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).write(hi32(mailbox))

    self.start_cpu(base)
    self.wait_cpu_halted(base)

    if mailbox is not None:
      return self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(base).read(), self.nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base(base).read()

  def disable_ctx_req(self, base:int):
    self.nvdev.NV_PFALCON_FBIF_CTL.with_base(base).update(allow_phys_no_ctx=1)
    self.nvdev.NV_PFALCON_FALCON_DMACTL.with_base(base).write(0x0)

  def reset(self, base:int, riscv=False):
    engine_reg = self.nvdev.NV_PGSP_FALCON_ENGINE if base == self.falcon else self.nvdev.NV_PSEC_FALCON_ENGINE
    engine_reg.write(reset=1)
    time.sleep(0.1)
    engine_reg.write(reset=0)

    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['mem_scrubbing'], value=0, msg="Scrubbing not completed")

    if riscv: self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=1, valid=0, brfetch=1)
    elif self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['riscv'] == 1:
      self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=0)
      wait_cond(lambda: self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).read_bitfields()['valid'], msg="RISCV core not booted")
      self.nvdev.NV_PFALCON_FALCON_RM.with_base(base).write(self.nvdev.chip_id)

class NV_FLCN_COT(NV_IP):
  def wait_for_reset(self):
    self.nvdev.include("dev_therm", "gb202")
    wait_cond(lambda _: self.nvdev.NV_THERM_I2CS_SCRATCH.read() == 0xff, "waiting for reset")

  def init_sw(self):
    self.nvdev.include("dev_gsp", "ga102")
    self.nvdev.include("dev_falcon_v4", "gh100")
    self.nvdev.include("dev_vm", "gh100")
    self.nvdev.include("dev_fsp_pri", "gh100")
    self.nvdev.include("dev_bus", "tu102")

    self.fmc_boot_args_view, _, fmc_boot_addrs = self.nvdev._alloc_boot_mem(ctypes.sizeof(nv.GSP_FMC_BOOT_PARAMS),
      data=bytes(nv.GSP_FMC_BOOT_PARAMS()))
    self.fmc_boot_args_sysmem = fmc_boot_addrs[0]
    self.init_fmc_image()

  def init_fmc_image(self):
    _, sections, _ = elf_loader(fetch_fw(f"nvidia/{self.nvdev.fw_name}/gsp", "fmc-570.144.bin",
                                         "cb59a35c1d4bd1274d7267fd10243c29f843ff41c851b9cbd59f5af2ddd7fece"))
    def _section(s): return next((sh.content for sh in sections if sh.name == s))
    self.fmc_booter_image, self.fmc_booter_hash = _section("image"), memoryview(_section("hash")).cast('I')
    self.fmc_booter_sig, self.fmc_booter_pkey = memoryview(_section("signature")).cast('I'), memoryview(_section("publickey") + b"\x00" * 3).cast('I')
    _, _, fmc_booter_addrs = self.nvdev._alloc_boot_mem(len(self.fmc_booter_image), data=self.fmc_booter_image)
    self.fmc_booter_bar1 = fmc_booter_addrs[0]

  def init_hw(self):
    self.falcon = 0x00110000

    boot_args = nv.GSP_ACR_BOOT_GSP_RM_PARAMS(gspRmDescOffset=self.nvdev.gsp.wpr_meta_sysmem,
      gspRmDescSize=ctypes.sizeof(nv.GspFwWprMeta), target=nv.GSP_DMA_TARGET_COHERENT_SYSTEM, bIsGspRmBoot=True)
    rm_args = nv.GSP_RM_PARAMS(bootArgsOffset=self.nvdev.gsp.libos_args_sysmem, target=nv.GSP_DMA_TARGET_COHERENT_SYSTEM)
    self.fmc_boot_args_view[:ctypes.sizeof(nv.GSP_FMC_BOOT_PARAMS)] = bytes(nv.GSP_FMC_BOOT_PARAMS(bootGspRmParams=boot_args, gspRmParams=rm_args))

    cot_payload = nv.NVDM_PAYLOAD_COT(version=0x2, size=ctypes.sizeof(nv.NVDM_PAYLOAD_COT), frtsVidmemOffset=0x1c00000, frtsVidmemSize=0x100000,
      gspBootArgsSysmemOffset=self.fmc_boot_args_sysmem, gspFmcSysmemOffset=self.fmc_booter_bar1)
    for i,x in enumerate(self.fmc_booter_hash): cot_payload.hash384[i] = x
    for i,x in enumerate(self.fmc_booter_sig): cot_payload.signature[i] = x
    for i,x in enumerate(self.fmc_booter_pkey): cot_payload.publicKey[i] = x

    self.kfsp_send_msg(nv.NVDM_TYPE_COT, bytes(cot_payload))
    wait_cond(lambda: self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(self.falcon).read_bitfields()['riscv_br_priv_lockdown'], value=0)

  def kfsp_send_msg(self, nvmd:int, buf:bytes):
    # All single-packets go to seid 0
    headers = int.to_bytes((1 << 31) | (1 << 30), 4, 'little') + int.to_bytes((0x7e << 0) | (0x10de << 8) | (nvmd << 24), 4, 'little')
    buf = headers + buf + (4 - (len(buf) % 4)) * b'\x00'
    assert len(buf) < 0x400, f"FSP message too long: {len(buf)} bytes, max 1024 bytes"

    self.nvdev.NV_PFSP_EMEMC[0].write(offs=0, blk=0, aincw=1, aincr=0)
    for i in range(0, len(buf), 4): self.nvdev.NV_PFSP_EMEMD[0].write(int.from_bytes(buf[i:i+4], 'little'))

    self.nvdev.NV_PFSP_QUEUE_TAIL[0].write(len(buf) - 4)
    self.nvdev.NV_PFSP_QUEUE_HEAD[0].write(0)

    # Waiting for a response
    wait_cond(lambda: self.nvdev.NV_PFSP_MSGQ_HEAD[0].read() != self.nvdev.NV_PFSP_MSGQ_TAIL[0].read(), msg="FSP didn't respond to message")

    self.nvdev.NV_PFSP_EMEMC[0].write(offs=0, blk=0, aincw=0, aincr=1)
    self.nvdev.NV_PFSP_MSGQ_TAIL[0].write(self.nvdev.NV_PFSP_MSGQ_HEAD[0].read())

class NV_GSP(NV_IP):
  def init_sw(self):
    self.handle_gen = itertools.count(0xcf000000)
    self.init_rm_args()
    self.init_libos_args()
    self.init_wpr_meta()

    # Prefill cmd queue with info for gsp to start.
    self.rpc_set_gsp_system_info()
    self.rpc_set_registry_table()

    self.gpfifo_class, self.compute_class, self.dma_class = nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_COMPUTE_B, nv_gpu.AMPERE_DMA_COPY_B
    match self.nvdev.chip_name[:2]:
      case "AD": self.compute_class = nv_gpu.ADA_COMPUTE_A
      case "GB":
        self.gpfifo_class,self.compute_class,self.dma_class=nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A,nv_gpu.BLACKWELL_COMPUTE_B,nv_gpu.BLACKWELL_DMA_COPY_B

  def init_rm_args(self, queue_size=0x40000):
    # Alloc queues
    pte_cnt = ((queue_pte_cnt:=(queue_size * 2) // 0x1000)) + round_up(queue_pte_cnt * 8, 0x1000) // 0x1000
    pt_size = round_up(pte_cnt * 8, 0x1000)
    queues_view, _, queues_sysmem = self.nvdev._alloc_boot_mem(pt_size + queue_size * 2, sysmem=True)

    # Fill up ptes
    for i, sysmem in enumerate(queues_sysmem): queues_view.view(i * 0x8, 0x8, fmt='Q')[0] = sysmem

    # Fill up arguments
    queue_args = nv.MESSAGE_QUEUE_INIT_ARGUMENTS(sharedMemPhysAddr=queues_sysmem[0], pageTableEntryCount=pte_cnt, cmdQueueOffset=pt_size,
      statQueueOffset=pt_size + queue_size)
    _, _, rm_args_addrs = self.nvdev._alloc_boot_mem(ctypes.sizeof(nv.GSP_ARGUMENTS_CACHED),
      data=bytes(nv.GSP_ARGUMENTS_CACHED(bDmemStack=True, messageQueueInitArguments=queue_args)))
    self.rm_args_sysmem = rm_args_addrs[0]

    # Build command queue header
    # self.cmd_q_va, self.stat_q_va = queues_view.addr + pt_size, queues_view.addr + pt_size + queue_size
    self.cmd_q_view, self.stat_q_view = queues_view.view(pt_size), queues_view.view(pt_size + queue_size)

    self.cmd_q_view[:ctypes.sizeof(nv.msgqTxHeader)] = bytes(nv.msgqTxHeader(version=0, size=queue_size, entryOff=0x1000, msgSize=0x1000,
      msgCount=(queue_size - 0x1000) // 0x1000, writePtr=0, flags=1, rxHdrOff=ctypes.sizeof(nv.msgqTxHeader)))

    self.cmd_q = NVRpcQueue(self, self.cmd_q_view, None)

  def init_libos_args(self):
    _, _, logbuf_addrs = self.nvdev._alloc_boot_mem(2 << 20)
    libos_args_view, _, libos_addrs = self.nvdev._alloc_boot_mem(0x1000)
    self.libos_args_sysmem = libos_addrs[0]

    libos_structs = [nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x10000,
        id8=int.from_bytes(bytes(f"LOG{name}", 'utf-8'), 'big'), pa=logbuf_addrs[0] + 0x10000 * i)
        for i, name in enumerate(["INIT", "INTR", "RM", "MNOC", "KRNL"])]
    libos_structs.append(nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x1000,
        id8=int.from_bytes(bytes("RMARGS", 'utf-8'), 'big'), pa=self.rm_args_sysmem))
    libos_args_view[:sum(ctypes.sizeof(s) for s in libos_structs)] = b''.join(bytes(s) for s in libos_structs)

  def init_gsp_image(self):
    _, sections, _ = elf_loader(fetch_fw("nvidia/ga102/gsp", "gsp-570.144.bin", "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f"))
    self.gsp_image = next((sh.content for sh in sections if sh.name == ".fwimage"))
    signature = next((sh.content for sh in sections if sh.name == (f".fwsignature_{self.nvdev.chip_name[:4].lower()}x")))

    # Build radix3
    npages = [0, 0, 0, round_up(len(self.gsp_image), 0x1000) // 0x1000]
    for i in range(3, 0, -1): npages[i-1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1

    offsets = [sum(npages[:i]) * 0x1000 for i in range(4)]
    radix_view, _, self.gsp_radix3_addrs = self.nvdev._alloc_boot_mem(offsets[-1] + len(self.gsp_image))

    # Copy image
    radix_view.view(offsets[-1], len(self.gsp_image))[:] = self.gsp_image

    # Copy level and image pages.
    for i in range(0, 3):
      cur_offset = sum(npages[:i+1])
      radix_view.view(offsets[i], npages[i+1] * 8, fmt='Q')[:] = array.array('Q', self.gsp_radix3_addrs[cur_offset:cur_offset+npages[i+1]])

    # Copy signature
    _, _, gsp_sig_addrs = self.nvdev._alloc_boot_mem(len(signature), data=signature)
    self.gsp_signature_bar1 = gsp_sig_addrs[0]

  def init_boot_binary_image(self):
    sha = {"ga102":"82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad",
           "ad102":"65ab2e6b6e0fca95365c4deac79a34582abcfeb15b6ae234138f22e7183118a8",
           "gb202":"d40b48e431d1707dc77af3605db358ed7a32ebfc2830eb74de2eddb4d3025071"}[self.nvdev.fw_name]
    h = nv.struct_nvfw_bin_hdr.from_buffer_copy(b:=fetch_fw(f"nvidia/{self.nvdev.fw_name}/gsp", "bootloader-570.144.bin", sha))
    self.booter_image, self.booter_desc = b[h.data_offset:h.data_offset+h.data_size], nv.RM_RISCV_UCODE_DESC.from_buffer_copy(b, h.header_offset)
    _, _, booter_addrs = self.nvdev._alloc_boot_mem(len(self.booter_image), data=self.booter_image)
    self.booter_bar1 = booter_addrs[0]

  def init_wpr_meta(self):
    self.init_gsp_image()
    self.init_boot_binary_image()

    common = {'sizeOfBootloader':(boot_sz:=len(self.booter_image)), 'sysmemAddrOfBootloader':self.booter_bar1,
      'sizeOfRadix3Elf':(radix3_sz:=len(self.gsp_image)), 'sysmemAddrOfRadix3Elf': self.gsp_radix3_addrs[0],
      'sizeOfSignature': 0x1000, 'sysmemAddrOfSignature': self.gsp_signature_bar1,
      'bootloaderCodeOffset': self.booter_desc.monitorCodeOffset, 'bootloaderDataOffset': self.booter_desc.monitorDataOffset,
      'bootloaderManifestOffset': self.booter_desc.manifestOffset, 'revision':nv.GSP_FW_WPR_META_REVISION, 'magic':nv.GSP_FW_WPR_META_MAGIC}

    if self.nvdev.fmc_boot:
      m = nv.GspFwWprMeta(**common, vgaWorkspaceSize=0x20000, pmuReservedSize=0x1820000, nonWprHeapSize=0x220000, gspFwHeapSize=0x8700000,
        frtsSize=0x100000)
    else:
      m = nv.GspFwWprMeta(**common, vgaWorkspaceSize=(vga_sz:=0x100000), vgaWorkspaceOffset=(vga_off:=self.nvdev.vram_size-vga_sz),
        gspFwWprEnd=vga_off, frtsSize=(frts_sz:=0x100000), frtsOffset=(frts_off:=vga_off-frts_sz), bootBinOffset=(boot_off:=frts_off-boot_sz),
        gspFwOffset=(gsp_off:=round_down(boot_off-radix3_sz, 0x10000)), gspFwHeapSize=(gsp_heap_sz:=0x8100000), fbSize=self.nvdev.vram_size,
        gspFwHeapOffset=(gsp_heap_off:=round_down(gsp_off-gsp_heap_sz, 0x100000)), gspFwWprStart=(wpr_st:=round_down(gsp_heap_off-0x1000, 0x100000)),
        nonWprHeapSize=(non_wpr_sz:=0x100000), nonWprHeapOffset=(non_wpr_off:=round_down(wpr_st-non_wpr_sz, 0x100000)), gspFwRsvdStart=non_wpr_off)
      assert self.nvdev.flcn.frts_offset == m.frtsOffset, f"FRTS mismatch: {self.nvdev.flcn.frts_offset} != {m.frtsOffset}"
    self.wpr_meta, _, wpr_meta_addrs = self.nvdev._alloc_boot_mem(ctypes.sizeof(type(m)), data=bytes(m))
    self.wpr_meta_sysmem = wpr_meta_addrs[0]

  def promote_ctx(self, client:int, subdevice:int, obj:int, ctxbufs:dict[int, GRBufDesc], bufs=None, virt=None, phys=None):
    res, prom = {}, nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS(entryCount=len(ctxbufs), engineType=0x1, hChanClient=client, hObject=obj)
    for i,(buf,desc) in enumerate(ctxbufs.items()):
      use_v, use_p = (desc.virt if virt is None else virt), (desc.phys if phys is None else phys)
      x = (bufs or {}).get(buf, self.nvdev.mm.valloc(desc.size, contiguous=True)) # allocate buffers
      prom.promoteEntry[i] = nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY(bufferId=buf, gpuVirtAddr=x.va_addr if use_v else 0, bInitialize=use_p,
        gpuPhysAddr=x.paddrs[0][0] if use_p else 0, size=desc.size if use_p else 0, physAttr=0x4 if use_p else 0, bNonmapped=(use_p and not use_v))
      res[buf] = x
    self.rpc_rm_control(hObject=subdevice, cmd=nv_gpu.NV2080_CTRL_CMD_GPU_PROMOTE_CTX, params=prom, client=client)
    return res

  def init_golden_image(self):
    self.rpc_rm_alloc(hParent=0x0, hClass=0x0, params=nv_gpu.NV0000_ALLOC_PARAMETERS())
    dev = self.rpc_rm_alloc(hParent=self.priv_root, hClass=nv_gpu.NV01_DEVICE_0, params=nv_gpu.NV0080_ALLOC_PARAMETERS(hClientShare=self.priv_root))
    subdev = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.NV20_SUBDEVICE_0, params=nv_gpu.NV2080_ALLOC_PARAMETERS())
    vaspace = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.FERMI_VASPACE_A, params=nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS())

    # reserve 512MB for the reserved PDES
    res_va = self.nvdev.mm.alloc_vaddr(res_sz:=(512 << 20))

    bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(pageSize=res_sz, numLevelsToCopy=3,
      virtAddrLo=res_va, virtAddrHi=res_va + res_sz - 1)
    for i,pt in enumerate(self.nvdev.mm.page_tables(res_va, size=res_sz)):
      bufs_p.levels[i] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_level(physAddress=pt.paddr,
        size=self.nvdev.mm.pte_cnt[0] * 8 if i == 0 else 0x1000, pageShift=self.nvdev.mm.pte_covers[i].bit_length() - 1, aperture=1)
    self.rpc_rm_control(hObject=vaspace, cmd=nv_gpu.NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES, params=bufs_p)

    gpfifo_area = self.nvdev.mm.valloc(4 << 10, contiguous=True)
    userd = nv_gpu.NV_MEMORY_DESC_PARAMS(base=gpfifo_area.paddrs[0][0] + 0x20 * 8, size=0x20, addressSpace=2, cacheAttrib=0)
    gg_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr, gpFifoEntries=32, engineType=0x1, cid=3,
      hVASpace=vaspace, userdOffset=(ctypes.c_uint64*8)(0x20 * 8), userdMem=userd, internalFlags=0x1a, flags=0x200320)
    ch_gpfifo = self.rpc_rm_alloc(hParent=dev, hClass=self.gpfifo_class, params=gg_params)

    gr_ctx_bufs_info = self.rpc_rm_control(hObject=subdev, cmd=nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO,
      params=nv_gpu.NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS()).engineContextBuffersInfo[0]
    def _ctx_info(idx, add=0, align=None): return round_up(gr_ctx_bufs_info.engine[idx].size + add, align or gr_ctx_bufs_info.engine[idx].alignment)

    # Setup graphics context
    gr_size = _ctx_info(nv_gpu.NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS, add=0x40000)
    patch_size = _ctx_info(nv_gpu.NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH)
    cfgs_sizes = {x: _ctx_info(x + 14, align=(2 << 20) if x == 5 else None) for x in range(3, 11)} # indices 3–10 are mapped to 17–24
    self.grctx_bufs = {0: GRBufDesc(gr_size, phys=True, virt=True), 1: GRBufDesc(patch_size, phys=True, virt=True, local=True),
      2: GRBufDesc(patch_size, phys=True, virt=True), **{x: GRBufDesc(cfgs_sizes[x], phys=False, virt=True) for x in range(3, 7)},
      9: GRBufDesc(cfgs_sizes[9], phys=True, virt=True), 10: GRBufDesc(cfgs_sizes[10], phys=True, virt=False),
      11: GRBufDesc(cfgs_sizes[10], phys=True, virt=True)} # NOTE: 11 reuses cfgs_sizes[10]
    self.promote_ctx(self.priv_root, subdev, ch_gpfifo, {k:v for k, v in self.grctx_bufs.items() if not v.local})

    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=self.compute_class, params=None)
    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=self.dma_class, params=None)

  def init_hw(self):
    self.stat_q = NVRpcQueue(self, self.stat_q_view, self.cmd_q_view)
    self.cmd_q.rx_view = self.stat_q_view.view(self.stat_q.tx.rxHdrOff, fmt='I')

    self.stat_q.wait_resp(nv.NV_VGPU_MSG_EVENT_GSP_INIT_DONE)

    self.nvdev.NV_PBUS_BAR1_BLOCK.write(mode=0, target=0, ptr=0)
    if self.nvdev.fmc_boot: self.nvdev.NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR.write(mode=0, target=0, ptr=0)

    self.priv_root = 0xc1e00004
    self.init_golden_image()

  def fini_hw(self): self.rpc_unloading_guest_driver()

  ### RPCs

  def rpc_alloc_memory(self, hDevice:int, hClass:int, paddrs:list[tuple[int,int]], length:int, flags:int, client:int|None=None) -> int:
    assert all(sz == 0x1000 for _, sz in paddrs), f"all pages must be 4KB, got {[(hex(p), hex(sz)) for p, sz in paddrs]}"

    rpc = nv.rpc_alloc_memory_v(hClient=(client:=client or self.priv_root), hDevice=hDevice, hMemory=(handle:=next(self.handle_gen)),
      hClass=hClass, flags=flags, pteAdjust=0, format=6, length=length, pageCount=len(paddrs))
    rpc.pteDesc.idr, rpc.pteDesc.length = nv.NV_VGPU_PTEDESC_IDR_NONE, (len(paddrs) & 0xffff)

    payload = bytes(rpc) + b''.join(bytes(nv.struct_pte_desc_pte_pde(pte=(paddr >> 12))) for paddr, _ in paddrs)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY, bytes(payload))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY)
    return handle

  def rpc_rm_alloc(self, hParent:int, hClass:int, params:Any, client=None) -> int:
    if hClass == self.gpfifo_class:
      ramfc_alloc = self.nvdev.mm.valloc(0x1000, contiguous=True)
      params.ramfcMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x200, addressSpace=2, cacheAttrib=0)
      params.instanceMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x1000, addressSpace=2, cacheAttrib=0)

      _, method_paddr, _ = self.nvdev._alloc_boot_mem(0x5000, sysmem=False)
      params.mthdbufMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=method_paddr, size=0x5000, addressSpace=2, cacheAttrib=0)

      if client is not None and client != self.priv_root and params.hObjectError != 0:
        params.errorNotifierMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=0, size=0xecc, addressSpace=0, cacheAttrib=0)
        params.userdMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=params.hUserdMemory[0] + params.userdOffset[0], size=0x400, addressSpace=2, cacheAttrib=0)

    alloc_args = nv.rpc_gsp_rm_alloc_v(hClient=(client:=client or self.priv_root), hParent=hParent, hObject=(obj:=next(self.handle_gen)),
      hClass=hClass, flags=0x0, paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, bytes(alloc_args) + (bytes(params) if params is not None else b''))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC)

    if hClass == nv_gpu.FERMI_VASPACE_A and client != self.priv_root:
      self.rpc_set_page_directory(device=hParent, hVASpace=obj, pdir_paddr=self.nvdev.mm.root_page_table.paddr, client=client)
    if hClass == nv_gpu.NV01_DEVICE_0 and client != self.priv_root: self.device = obj # save user device handle
    if hClass == nv_gpu.NV20_SUBDEVICE_0: self.subdevice = obj # save subdevice handle
    if hClass == self.compute_class and client != self.priv_root:
      phys_gr_ctx = self.promote_ctx(client, self.subdevice, hParent, {k:v for k,v in self.grctx_bufs.items() if k in [0, 1, 2]}, virt=False)
      self.promote_ctx(client, self.subdevice, hParent, {k:v for k,v in self.grctx_bufs.items() if k in [0, 1, 2]}, phys_gr_ctx, phys=False)
    return obj if hClass != nv_gpu.NV1_ROOT else client

  def rpc_rm_control(self, hObject:int, cmd:int, params:Any, client=None, extra=None):
    if cmd == nv_gpu.NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES:
      self.rpc_rm_control(hObject, nv_gpu.NVB0CC_CTRL_CMD_INTERNAL_PERMISSIONS_INIT, nv_gpu.NVB0CC_CTRL_INTERNAL_PERMISSIONS_INIT_PARAMS(
        bAdminProfilingPermitted=1, bDevProfilingPermitted=1, bCtxProfilingPermitted=1, bVideoMemoryProfilingPermitted=1,
        bSysMemoryProfilingPermitted=1), client=client)
    elif cmd == nv_gpu.NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM:
      params.hMemPmaBuffer = self.rpc_alloc_memory(self.device, nv_gpu.NV01_MEMORY_LIST_SYSTEM, extra[0].meta.mapping.paddrs, extra[0].size,
        pma_flags:=(nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4 | nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30), client=client)
      params.hMemPmaBytesAvailable = self.rpc_alloc_memory(self.device, nv_gpu.NV01_MEMORY_LIST_SYSTEM, extra[1].meta.mapping.paddrs, extra[1].size,
        pma_flags | nv_gpu.NVOS02_FLAGS_ALLOC_USER_READ_ONLY_YES << 21, client=client)

    control_args = nv.rpc_gsp_rm_control_v(hClient=(client:=client or self.priv_root), hObject=hObject, cmd=cmd, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, bytes(control_args) + (bytes(params) if params is not None else b''))
    res = self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL)
    st = type(params).from_buffer_copy(res[len(bytes(control_args)):]) if params is not None else None

    # NOTE: gb20x requires the enable bit for token submission. Patch workSubmitToken here to maintain userspace compatibility.
    if self.nvdev.chip_name.startswith("GB2") and cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
      cast(nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS, st).workSubmitToken |= (1 << 30)
    return st

  def rpc_set_page_directory(self, device:int, hVASpace:int, pdir_paddr:int, client=None, pasid=0xffffffff):
    params = nv.struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05(physAddress=pdir_paddr,
      numEntries=self.nvdev.mm.pte_cnt[0], flags=0x8, hVASpace=hVASpace, pasid=pasid, subDeviceId=1, chId=0) # flags field is all channels.
    alloc_args = nv.rpc_set_page_directory_v(hClient=client or self.priv_root, hDevice=device, pasid=pasid, params=params)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, bytes(alloc_args))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY)

  def rpc_set_gsp_system_info(self):
    def bdf_as_int(s): return 0x000 if s.startswith("usb") or s.startswith("remote") else (int(s[5:7],16)<<8) | (int(s[8:10],16)<<3) | int(s[-1],16)

    pcidev = self.nvdev.pci_dev
    data = nv.GspSystemInfo(gpuPhysAddr=pcidev.bar_info(0)[0], gpuPhysFbAddr=pcidev.bar_info(1)[0], gpuPhysInstAddr=pcidev.bar_info(3)[0],
      pciConfigMirrorBase=[0x88000, 0x92000][self.nvdev.fmc_boot], pciConfigMirrorSize=0x1000, nvDomainBusDeviceFunc=bdf_as_int(self.nvdev.devfmt),
      bIsPassthru=1, PCIDeviceID=pcidev.read_config(pci.PCI_VENDOR_ID, 4), PCISubDeviceID=pcidev.read_config(pci.PCI_SUBSYSTEM_VENDOR_ID, 4),
      PCIRevisionID=pcidev.read_config(pci.PCI_REVISION_ID, 1), maxUserVa=0x7ffffffff000)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, bytes(data))

  def rpc_unloading_guest_driver(self):
    data = nv.rpc_unloading_guest_driver_v(bInPMTransition=0, bGc6Entering=0, newLevel=(__GPU_STATE_FLAGS_FAST_UNLOAD:=1 << 6))
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER, bytes(data))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER)

  def rpc_set_registry_table(self):
    table = {'RMForcePcieConfigSave': 0x1, 'RMSecBusResetEnable': 0x1}
    entries_bytes, data_bytes = bytes(), bytes()
    hdr_size, entries_size = ctypes.sizeof(nv.PACKED_REGISTRY_TABLE), ctypes.sizeof(nv.PACKED_REGISTRY_ENTRY) * len(table)

    for k,v in table.items():
      entries_bytes += bytes(nv.PACKED_REGISTRY_ENTRY(nameOffset=hdr_size + entries_size + len(data_bytes),
                                                      type=nv.REGISTRY_TABLE_ENTRY_TYPE_DWORD, data=v, length=4))
      data_bytes += k.encode('utf-8') + b'\x00'

    header = nv.PACKED_REGISTRY_TABLE(size=hdr_size + len(entries_bytes) + len(data_bytes), numEntries=len(table))
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_REGISTRY, bytes(header) + entries_bytes + data_bytes)

  def run_cpu_seq(self, seq_buf:bytes):
    hdr = nv.rpc_run_cpu_sequencer_v17_00.from_buffer_copy(seq_buf[:(hdr_sz:=ctypes.sizeof(nv.rpc_run_cpu_sequencer_v17_00))])
    cmd_iter = iter(memoryview(seq_buf[hdr_sz:]).cast('I')[:hdr.cmdIndex])

    for op in cmd_iter:
      if op == 0x0: self.nvdev.wreg(next(cmd_iter), next(cmd_iter)) # reg write
      elif op == 0x1: # reg modify
        addr, val, mask = next(cmd_iter), next(cmd_iter), next(cmd_iter)
        self.nvdev.wreg(addr, (self.nvdev.rreg(addr) & ~mask) | (val & mask))
      elif op == 0x2: # reg poll
        addr, mask, val, _, _ = next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter)
        wait_cond(lambda a, m: (self.nvdev.rreg(a) & m), addr, mask, value=val, msg=f"Register {addr:#x} not equal to {val:#x} after polling")
      elif op == 0x3: time.sleep(next(cmd_iter) / 1e6) # delay us
      elif op == 0x4: # save reg
        addr, index = next(cmd_iter), next(cmd_iter)
        hdr.regSaveArea[index] = self.nvdev.rreg(addr)
      elif op == 0x5: # core reset
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon)
        self.nvdev.flcn.disable_ctx_req(self.nvdev.flcn.falcon)
      elif op == 0x6: self.nvdev.flcn.start_cpu(self.nvdev.flcn.falcon)
      elif op == 0x7: self.nvdev.flcn.wait_cpu_halted(self.nvdev.flcn.falcon)
      elif op == 0x8: # core resume
        self.nvdev.flcn.reset(self.nvdev.flcn.falcon, riscv=True)

        self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.libos_args_sysmem))
        self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.libos_args_sysmem))

        self.nvdev.flcn.start_cpu(self.nvdev.flcn.sec2)
        wait_cond(lambda: self.nvdev.NV_PGC6_BSI_SECURE_SCRATCH_14.read_bitfields()['boot_stage_3_handoff'], msg="SEC2 didn't hand off")

        mailbox = self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(self.nvdev.flcn.sec2).read()
        assert mailbox == 0x0, f"Falcon SEC2 failed to execute, mailbox is {mailbox:08x}"
      else: raise ValueError(f"Unknown op code {op} in run_cpu_seq")
