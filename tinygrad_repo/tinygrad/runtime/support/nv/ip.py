from __future__ import annotations
import ctypes, time, array, struct, itertools, dataclasses
from tinygrad.runtime.autogen.nv import nv
from tinygrad.helpers import to_mv, lo32, hi32, DEBUG, round_up, round_down, mv_address, fetch
from tinygrad.runtime.support.system import System
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import nv_gpu

@dataclasses.dataclass(frozen=True)
class GRBufDesc: size:int; v:int; p:int; lc:int=0 # noqa: E702

class NV_IP:
  def __init__(self, nvdev): self.nvdev = nvdev
  def init_sw(self): pass # Prepare sw/allocations for this IP
  def init_hw(self): pass # Initialize hw for this IP

class NVRpcQueue:
  def __init__(self, gsp:NV_GSP, va:int, completion_q_va:int|None=None):
    self.tx = nv.msgqTxHeader.from_address(va)
    while self.tx.entryOff != 0x1000: pass # wait for the tx header to be initialized

    if completion_q_va is not None: self.rx = nv.msgqRxHeader.from_address(completion_q_va + nv.msgqTxHeader.from_address(completion_q_va).rxHdrOff)

    self.gsp, self.va, self.queue_va, self.seq = gsp, va, va + self.tx.entryOff, 0
    self.queue_mv = to_mv(self.queue_va, self.tx.msgSize * self.tx.msgCount)

  def _checksum(self, data):
    if (pad_len:=(-len(data)) % 8): data += b'\x00' * pad_len
    checksum = 0
    for offset in range(0, len(data), 8): checksum ^= struct.unpack_from('Q', data, offset)[0]
    return ((checksum >> 32) & 0xFFFFFFFF) ^ (checksum & 0xFFFFFFFF)

  def send_rpc(self, func, msg, wait=False):
    header = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, rpc_result=nv.NV_VGPU_MSG_RESULT_RPC_PENDING,
      rpc_result_private=nv.NV_VGPU_MSG_RESULT_RPC_PENDING, header_version=(3<<24), function=func, length=len(msg) + 0x20)

    msg = bytes(header) + msg
    phdr = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=round_up(len(msg), self.tx.msgSize) // self.tx.msgSize, seqNum=self.seq)
    phdr.checkSum = self._checksum(bytes(phdr) + msg)
    msg = bytes(phdr) + msg

    off = self.tx.writePtr * self.tx.msgSize
    self.queue_mv[off:off+len(msg)] = msg
    self.tx.writePtr = (self.tx.writePtr + round_up(len(msg), self.tx.msgSize) // self.tx.msgSize) % self.tx.msgCount
    System.memory_barrier()

    self.seq += 1
    self.gsp.nvdev.NV_PGSP_QUEUE_HEAD[0].write(0x0)

  def wait_resp(self, cmd) -> memoryview:
    while True:
      System.memory_barrier()
      if self.rx.readPtr == self.tx.writePtr: continue

      off = self.rx.readPtr * self.tx.msgSize
      hdr = nv.rpc_message_header_v.from_address(self.queue_va + off + 0x30)
      msg = self.queue_mv[off + 0x50 : off + 0x50 + hdr.length]

      # Handling special functions
      if hdr.function == nv.NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER: self.gsp.run_cpu_seq(msg)
      elif hdr.function == nv.NV_VGPU_MSG_EVENT_OS_ERROR_LOG: print(f"GSP LOG: {msg[12:].tobytes().rstrip(bytes([0])).decode('utf-8')}")

      # Update the read pointer
      self.rx.readPtr = (self.rx.readPtr + round_up(hdr.length, self.tx.msgSize) // self.tx.msgSize) % self.tx.msgCount
      System.memory_barrier()

      if DEBUG >= 2:
        rpc_names = {**nv.c__Ea_NV_VGPU_MSG_FUNCTION_NOP__enumvalues, **nv.c__Ea_NV_VGPU_MSG_EVENT_FIRST_EVENT__enumvalues}
        print(f"nv {self.gsp.nvdev.devfmt}: in RPC: {rpc_names.get(hdr.function, f'ev:{hdr.function:x}')}, res:{hdr.rpc_result:#x}")

      if hdr.rpc_result != 0: raise RuntimeError(f"RPC call {hdr.function} failed with result {hdr.rpc_result}")
      if hdr.function == cmd: return msg

class NV_FLCN(NV_IP):
  def init_sw(self):
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_gsp.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_v4.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_v4_addendum.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_riscv_pri.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_fbif_v4.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_falcon_second_pri.h")
    self.nvdev.include("src/common/inc/swref/published/ampere/ga102/dev_sec_pri.h")
    self.nvdev.include("src/common/inc/swref/published/turing/tu102/dev_bus.h")

    self.prep_ucode()
    self.prep_booter()

  def prep_ucode(self):
    expansion_rom_off, bit_addr = 0x14e00, 0x1b0
    vbios_bytes = bytes(array.array('I', self.nvdev.mmio[0x00300000//4:(0x00300000+0x98e00)//4]))

    bit_header = nv.BIT_HEADER_V1_00.from_buffer_copy(vbios_bytes[bit_addr:bit_addr + ctypes.sizeof(nv.BIT_HEADER_V1_00)])
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

      # Patch image
      dmem = nv.FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3.from_buffer_copy(image[(dmem_mapper_offset:=self.desc_v3.IMEMLoadSize+0xae0):])
      dmem.init_cmd = cmd_id
      patched_image[dmem_mapper_offset : dmem_mapper_offset+len(bytes(dmem))] = bytes(dmem)
      patched_image[(cmd_off:=self.desc_v3.IMEMLoadSize+dmem.cmd_in_buffer_offset) : cmd_off+len(cmd)] = cmd
      patched_image[(sig_off:=self.desc_v3.IMEMLoadSize+self.desc_v3.PKCDataOffset) : sig_off+0x180] = signature[0x180:]

      return System.alloc_sysmem(len(patched_image), contiguous=True, data=patched_image)

    self.frts_image_va, self.frts_image_sysmem = __patch(0x15, bytes(frts_cmd))

  def prep_booter(self):
    image = self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "image_prod_data")
    sig = self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "sig_prod_data")
    header = self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "header_prod_data")
    patch_loc = int.from_bytes(self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "patch_loc_data"), 'little')
    sig_len = len(sig) // int.from_bytes(self.nvdev.extract_fw("kgspBinArchiveBooterLoadUcode", "num_sigs_data"), 'little')

    patched_image = bytearray(image)
    patched_image[patch_loc:patch_loc+sig_len] = sig[:sig_len]
    self.booter_image_va, self.booter_image_sysmem = System.alloc_sysmem(len(patched_image), contiguous=True, data=patched_image)
    _, _, self.booter_data_off, self.booter_data_sz, _, self.booter_code_off, self.booter_code_sz, _, _ = struct.unpack("9I", header)

  def init_hw(self):
    self.falcon, self.sec2 = 0x00110000, 0x00840000

    self.reset(self.falcon)
    self.execute_hs(self.falcon, self.frts_image_sysmem[0], code_off=0x0, data_off=self.desc_v3.IMEMLoadSize,
      imemPa=self.desc_v3.IMEMPhysBase, imemVa=self.desc_v3.IMEMVirtBase, imemSz=self.desc_v3.IMEMLoadSize,
      dmemPa=self.desc_v3.DMEMPhysBase, dmemVa=0x0, dmemSz=self.desc_v3.DMEMLoadSize,
      pkc_off=self.desc_v3.PKCDataOffset, engid=self.desc_v3.EngineIdMask, ucodeid=self.desc_v3.UcodeId)

    self.reset(self.falcon, riscv=True)

    # set up the mailbox
    self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.nvdev.gsp.libos_args_sysmem[0]))
    self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.nvdev.gsp.libos_args_sysmem[0]))

    # booter
    self.reset(self.sec2)
    mbx = self.execute_hs(self.sec2, self.booter_image_sysmem[0], code_off=self.booter_code_off, data_off=self.booter_data_off,
      imemPa=0x0, imemVa=self.booter_code_off, imemSz=self.booter_code_sz, dmemPa=0x0, dmemVa=0x0, dmemSz=self.booter_data_sz,
      pkc_off=0x10, engid=1, ucodeid=3, mailbox=self.nvdev.gsp.wpr_meta_sysmem)
    assert mbx[0] == 0x0, f"Booster failed to execute, mailbox is {mbx[0]:08x}, {mbx[1]:08x}"

    self.nvdev.NV_PFALCON_FALCON_OS.with_base(self.falcon).write(0x0)
    assert self.nvdev.NV_PRISCV_RISCV_CPUCTL.with_base(self.falcon).read_bitfields()['active_stat'] == 1, "GSP Core is not active"

  def execute_dma(self, base, cmd, dest, mem_off, sysmem, size):
    while self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'] != 0: pass

    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE.with_base(base).write(lo32(sysmem >> 8))
    self.nvdev.NV_PFALCON_FALCON_DMATRFBASE1.with_base(base).write(hi32(sysmem >> 8) & 0x1ff)

    xfered = 0
    while xfered < size:
      while self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['full'] != 0: pass

      self.nvdev.NV_PFALCON_FALCON_DMATRFMOFFS.with_base(base).write(dest + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFFBOFFS.with_base(base).write(mem_off + xfered)
      self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).write(cmd)
      xfered += 256

    while self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).read_bitfields()['idle'] != 1: pass

  def start_cpu(self, base):
    if self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['alias_en'] == 1:
      self.nvdev.wreg(base + self.nvdev.NV_PFALCON_FALCON_CPUCTL_ALIAS, 0x2)
    else: self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).write(startcpu=1)

  def wait_cpu_halted(self, base):
    while self.nvdev.NV_PFALCON_FALCON_CPUCTL.with_base(base).read_bitfields()['halted'] == 0: pass

  def execute_hs(self, base, img_sysmem, code_off, data_off, imemPa, imemVa, imemSz, dmemPa, dmemVa, dmemSz, pkc_off, engid, ucodeid, mailbox=None):
    self.disable_ctx_req(base)

    self.nvdev.NV_PFALCON_FBIF_TRANSCFG.with_base(base)[ctx_dma:=0].update(target=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM,
      mem_type=self.nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=1, sec=1)
    self.execute_dma(base, cmd, dest=imemPa, mem_off=imemVa, sysmem=img_sysmem+code_off-imemVa, size=imemSz)

    cmd = self.nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base(base).encode(write=0, size=self.nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B,
      ctxdma=ctx_dma, imem=0, sec=0)
    self.execute_dma(base, cmd, dest=dmemPa, mem_off=dmemVa, sysmem=img_sysmem+data_off-dmemVa, size=dmemSz)

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

  def disable_ctx_req(self, base):
    self.nvdev.NV_PFALCON_FBIF_CTL.with_base(base).update(allow_phys_no_ctx=1)
    self.nvdev.NV_PFALCON_FALCON_DMACTL.with_base(base).write(0x0)

  def reset(self, base, riscv=False):
    engine_reg = self.nvdev.NV_PGSP_FALCON_ENGINE if base == self.falcon else self.nvdev.NV_PSEC_FALCON_ENGINE
    engine_reg.write(reset=1)
    time.sleep(0.1)
    engine_reg.write(reset=0)

    while self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['mem_scrubbing'] != 0: time.sleep(0.1)

    if riscv: self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=1, valid=0, brfetch=1)
    elif self.nvdev.NV_PFALCON_FALCON_HWCFG2.with_base(base).read_bitfields()['riscv'] == 1:
      self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).write(core_select=0)
      while self.nvdev.NV_PRISCV_RISCV_BCR_CTRL.with_base(base).read_bitfields()['valid'] != 1: pass
      self.nvdev.NV_PFALCON_FALCON_RM.with_base(base).write(self.nvdev.chip_id)

class NV_GSP(NV_IP):
  def init_sw(self):
    self.handle_gen = itertools.count(0xcf000000)
    self.init_rm_args()
    self.init_libos_args()
    self.init_wpr_meta()

    # Prefill cmd queue with info for gsp to start.
    self.rpc_set_gsp_system_info()
    self.rpc_set_registry_table()

  def init_rm_args(self, queue_size=0x40000):
    # Alloc queues
    pte_cnt = ((queue_pte_cnt:=(queue_size * 2) // 0x1000)) + round_up(queue_pte_cnt * 8, 0x1000) // 0x1000
    pt_size = round_up(pte_cnt * 8, 0x1000)
    queues_va, queues_sysmem = System.alloc_sysmem(pt_size + queue_size * 2, contiguous=False)

    # Fill up ptes
    for i, sysmem in enumerate(queues_sysmem): to_mv(queues_va + i * 0x8, 0x8).cast('Q')[0] = sysmem

    # Fill up arguments
    queue_args = nv.MESSAGE_QUEUE_INIT_ARGUMENTS(sharedMemPhysAddr=queues_sysmem[0], pageTableEntryCount=pte_cnt, cmdQueueOffset=pt_size,
      statQueueOffset=pt_size + queue_size)
    rm_args, self.rm_args_sysmem = self.nvdev._alloc_boot_struct(nv.GSP_ARGUMENTS_CACHED(bDmemStack=True, messageQueueInitArguments=queue_args))

    # Build command queue header
    self.cmd_q_va, self.stat_q_va = queues_va + pt_size, queues_va + pt_size + queue_size

    cmd_q_tx = nv.msgqTxHeader(version=0, size=queue_size, entryOff=0x1000, msgSize=0x1000, msgCount=(queue_size - 0x1000) // 0x1000,
      writePtr=0, flags=1, rxHdrOff=ctypes.sizeof(nv.msgqTxHeader))
    to_mv(self.cmd_q_va, ctypes.sizeof(nv.msgqTxHeader))[:] = bytes(cmd_q_tx)

    self.cmd_q = NVRpcQueue(self, self.cmd_q_va, None)

  def init_libos_args(self):
    _, logbuf_sysmem = System.alloc_sysmem((2 << 20), contiguous=True)
    libos_args_va, self.libos_args_sysmem = System.alloc_sysmem(0x1000, contiguous=True)

    libos_structs = (nv.LibosMemoryRegionInitArgument * 6).from_address(libos_args_va)
    for i, name in enumerate(["INIT", "INTR", "RM", "MNOC", "KRNL"]):
      libos_structs[i] = nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x10000,
        id8=int.from_bytes(bytes(f"LOG{name}", 'utf-8'), 'big'), pa=logbuf_sysmem[0] + 0x10000 * i)

    libos_structs[5] = nv.LibosMemoryRegionInitArgument(kind=nv.LIBOS_MEMORY_REGION_CONTIGUOUS, loc=nv.LIBOS_MEMORY_REGION_LOC_SYSMEM, size=0x1000,
        id8=int.from_bytes(bytes("RMARGS", 'utf-8'), 'big'), pa=self.rm_args_sysmem)

  def init_gsp_image(self):
    fw = fetch("https://github.com/NVIDIA/linux-firmware/raw/refs/heads/nvidia-staging/nvidia/ga102/gsp/gsp-570.144.bin", subdir="fw").read_bytes()

    _, sections, _ = elf_loader(fw)
    self.gsp_image = next((sh.content for sh in sections if sh.name == ".fwimage"))
    signature = next((sh.content for sh in sections if sh.name == (f".fwsignature_{self.nvdev.chip_name[:4].lower()}x")))

    # Build radix3
    npages = [0, 0, 0, round_up(len(self.gsp_image), 0x1000) // 0x1000]
    for i in range(3, 0, -1): npages[i-1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1

    offsets = [sum(npages[:i]) * 0x1000 for i in range(4)]
    radix_va, self.gsp_radix3_sysmem = System.alloc_sysmem(offsets[-1] + len(self.gsp_image), contiguous=False)

    # Copy image
    to_mv(radix_va + offsets[-1], len(self.gsp_image))[:] = self.gsp_image

    # Copy level and image pages.
    for i in range(0, 3):
      cur_offset = sum(npages[:i+1])
      to_mv(radix_va + offsets[i], npages[i+1] * 8).cast('Q')[:] = array.array('Q', self.gsp_radix3_sysmem[cur_offset:cur_offset+npages[i+1]])

    # Copy signature
    self.gsp_signature_va, self.gsp_signature_sysmem = System.alloc_sysmem(len(signature), contiguous=True, data=signature)

  def init_boot_binary_image(self):
    self.booter_image = self.nvdev.extract_fw("kgspBinArchiveGspRmBoot", "ucode_image_prod_data")
    self.booter_desc = nv.RM_RISCV_UCODE_DESC.from_buffer_copy(self.nvdev.extract_fw("kgspBinArchiveGspRmBoot", "ucode_desc_prod_data"))
    _, self.booter_sysmem = System.alloc_sysmem(len(self.booter_image), contiguous=True, data=self.booter_image)

  def init_wpr_meta(self):
    self.init_gsp_image()
    self.init_boot_binary_image()

    m = nv.GspFwWprMeta(revision=nv.GSP_FW_WPR_META_REVISION, magic=nv.GSP_FW_WPR_META_MAGIC,
      fbSize=self.nvdev.vram_size, vgaWorkspaceSize=(vga_sz:=0x100000), vgaWorkspaceOffset=(vga_off:=self.nvdev.vram_size-vga_sz),
      sizeOfRadix3Elf=(radix3_sz:=len(self.gsp_image)), gspFwWprEnd=vga_off, frtsSize=(frts_sz:=0x100000), frtsOffset=(frts_off:=vga_off-frts_sz),
      sizeOfBootloader=(boot_sz:=len(self.booter_image)), bootBinOffset=(boot_off:=frts_off-boot_sz),
      gspFwOffset=(gsp_off:=round_down(boot_off-radix3_sz, 0x10000)), gspFwHeapSize=(gsp_heap_sz:=0x8100000),
      gspFwHeapOffset=(gsp_heap_off:=round_down(gsp_off-gsp_heap_sz, 0x100000)), gspFwWprStart=(wpr_start:=round_down(gsp_heap_off-0x1000, 0x100000)),
      nonWprHeapSize=(non_wpr_sz:=0x100000), nonWprHeapOffset=(non_wpr_off:=round_down(wpr_start-non_wpr_sz, 0x100000)), gspFwRsvdStart=non_wpr_off,
      sysmemAddrOfRadix3Elf=self.gsp_radix3_sysmem[0],sysmemAddrOfBootloader=self.booter_sysmem[0],sysmemAddrOfSignature=self.gsp_signature_sysmem[0],
      bootloaderCodeOffset=self.booter_desc.monitorCodeOffset, bootloaderDataOffset=self.booter_desc.monitorDataOffset,
      bootloaderManifestOffset=self.booter_desc.manifestOffset, sizeOfSignature=0x1000)
    self.wpr_meta, self.wpr_meta_sysmem = self.nvdev._alloc_boot_struct(m)
    assert self.nvdev.flcn.frts_offset == self.wpr_meta.frtsOffset, f"FRTS mismatch: {self.nvdev.flcn.frts_offset} != {self.wpr_meta.frtsOffset}"

  def promote_ctx(self, client, subdevice, obj, ctxbufs, bufs=None, virt=None, phys=None):
    res, prom = {}, nv_gpu.NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS(entryCount=len(ctxbufs), engineType=0x1, hChanClient=client, hObject=obj)
    for i,(buf,desc) in enumerate(ctxbufs.items()):
      use_v, use_p = (desc.v if virt is None else virt), (desc.p if phys is None else phys)
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
    res_va = self.nvdev.mm.alloc_vaddr(512 << 20)

    bufs_p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS(pageSize=(512<<20), numLevelsToCopy=3,
      virtAddrLo=res_va, virtAddrHi=res_va+(512<<20)-1)
    for i,pt in enumerate(self.nvdev.mm.page_tables(res_va, size=512<<20)[:3]):
      bufs_p.levels[i] = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_0(physAddress=pt.paddr, size=0x20 if i == 0 else 0x1000,
        pageShift=47 - (9 * i), aperture=1)
    self.rpc_rm_control(hObject=vaspace, cmd=0x90f10106, params=bufs_p)

    gpfifo_area = self.nvdev.mm.valloc(2<<20, contiguous=True)
    userd = nv_gpu.NV_MEMORY_DESC_PARAMS(base=gpfifo_area.paddrs[0][0] + 0x400 * 8, size=0x400, addressSpace=2, cacheAttrib=0)
    gg_params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr, gpFifoEntries=32, engineType=0x1, cid=3,
      hVASpace=vaspace, userdOffset=(ctypes.c_uint64*8)(0x400 * 8), userdMem=userd, internalFlags=0x1a, flags=0x200320)
    ch_gpfifo = self.rpc_rm_alloc(hParent=dev, hClass=nv_gpu.AMPERE_CHANNEL_GPFIFO_A, params=gg_params)

    self.grctx_bufs = {0: GRBufDesc(0x237000, p=1, v=1), 1: GRBufDesc(0x6000, p=1, v=1, lc=1), 2: GRBufDesc(0x6000, p=1, v=1),
      3: GRBufDesc(0x3000, p=0, v=1), 4: GRBufDesc(0x20000, p=0, v=1), 5: GRBufDesc(0x2600000, p=0, v=1), 6: GRBufDesc(0x80000, p=0, v=1),
      9: GRBufDesc(0x10000, p=1, v=1), 10: GRBufDesc(0x80000, p=1, v=0), 11: GRBufDesc(0x80000, p=1, v=1)}

    self.promote_ctx(self.priv_root, subdev, ch_gpfifo, {k:v for k, v in self.grctx_bufs.items() if v.lc == 0})

    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=nv_gpu.ADA_COMPUTE_A, params=None)
    self.rpc_rm_alloc(hParent=ch_gpfifo, hClass=nv_gpu.AMPERE_DMA_COPY_B, params=None)

  def init_hw(self):
    self.stat_q = NVRpcQueue(self, self.stat_q_va, self.cmd_q_va)
    self.cmd_q.rx = nv.msgqRxHeader.from_address(self.stat_q.va + self.stat_q.tx.rxHdrOff)

    self.stat_q.wait_resp(nv.NV_VGPU_MSG_EVENT_GSP_INIT_DONE)

    self.nvdev.NV_PBUS_BAR1_BLOCK.write(mode=0, target=0, ptr=0)
    self.priv_root = 0xc1e00004
    self.init_golden_image()

  ### RPCs

  def rpc_rm_alloc(self, hParent, hClass, params, client=None) -> int:
    if hClass == nv_gpu.AMPERE_CHANNEL_GPFIFO_A:
      ramfc_alloc = self.nvdev.mm.valloc(0x1000, contiguous=True)
      params.ramfcMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x200, addressSpace=2, cacheAttrib=0)
      params.instanceMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=ramfc_alloc.paddrs[0][0], size=0x1000, addressSpace=2, cacheAttrib=0)

      method_va, method_sysmem = System.alloc_sysmem(0x5000, contiguous=True)
      params.mthdbufMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=method_sysmem[0], size=0x5000, addressSpace=1, cacheAttrib=0)

      if client is not None and client != self.priv_root and params.hObjectError != 0:
        params.errorNotifierMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=0, size=0xecc, addressSpace=0, cacheAttrib=0)
        params.userdMem = nv_gpu.NV_MEMORY_DESC_PARAMS(base=params.hUserdMemory[0] + params.userdOffset[0], size=0x400, addressSpace=2, cacheAttrib=0)

    alloc_args = nv.rpc_gsp_rm_alloc_v(hClient=(client:=client or self.priv_root), hParent=hParent, hObject=(obj:=next(self.handle_gen)),
      hClass=hClass, flags=0x0, paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC, bytes(alloc_args) + (bytes(params) if params is not None else b''))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC)

    if hClass == nv_gpu.FERMI_VASPACE_A and client != self.priv_root:
      self.rpc_set_page_directory(device=hParent, hVASpace=obj, pdir_paddr=self.nvdev.mm.root_page_table.paddr, client=client)
    if hClass == nv_gpu.NV20_SUBDEVICE_0: self.subdevice = obj # save subdevice handle
    if hClass == nv_gpu.ADA_COMPUTE_A and client != self.priv_root:
      phys_gr_ctx = self.promote_ctx(client, self.subdevice, hParent, {k:v for k,v in self.grctx_bufs.items() if k in [0, 1, 2]}, virt=False)
      self.promote_ctx(client, self.subdevice, hParent, {k:v for k,v in self.grctx_bufs.items() if k in [0, 1, 2]}, phys_gr_ctx, phys=False)
    return obj if hClass != nv_gpu.NV1_ROOT else client

  def rpc_rm_control(self, hObject, cmd, params, client=None):
    control_args = nv.rpc_gsp_rm_control_v(hClient=(client:=client or self.priv_root), hObject=hObject, cmd=cmd, flags=0x0,
      paramsSize=ctypes.sizeof(params) if params is not None else 0x0)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL, bytes(control_args) + (bytes(params) if params is not None else b''))
    return self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL)[len(bytes(control_args)):]

  def rpc_set_page_directory(self, device, hVASpace, pdir_paddr, client=None):
    params = nv.struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05(physAddress=pdir_paddr,
      numEntries=self.nvdev.mm.pte_cnt[0], flags=0x8, hVASpace=hVASpace, pasid=0xffffffff, subDeviceId=1, chId=0) # flags field is all channels.
    alloc_args = nv.rpc_set_page_directory_v(hClient=client or self.priv_root, hDevice=device, pasid=0xffffffff, params=params)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY, bytes(alloc_args))
    self.stat_q.wait_resp(nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY)

  def rpc_set_gsp_system_info(self):
    def bdf_as_int(s): return (int(s[5:7],16)<<8) | (int(s[8:10],16)<<3) | int(s[-1],16)

    data = nv.GspSystemInfo(gpuPhysAddr=self.nvdev.bars[0][0], gpuPhysFbAddr=self.nvdev.bars[1][0], gpuPhysInstAddr=self.nvdev.bars[3][0],
      pciConfigMirrorBase=0x88000, pciConfigMirrorSize=0x1000, nvDomainBusDeviceFunc=bdf_as_int(self.nvdev.devfmt), bIsPassthru=1,
      PCIDeviceID=self.nvdev.venid, PCISubDeviceID=self.nvdev.subvenid, PCIRevisionID=self.nvdev.rev, maxUserVa=0x7ffffffff000)
    self.cmd_q.send_rpc(nv.NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, bytes(data))

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

  def run_cpu_seq(self, seq_buf):
    hdr = nv.rpc_run_cpu_sequencer_v17_00.from_address(mv_address(seq_buf))
    cmd_iter = iter(seq_buf[ctypes.sizeof(nv.rpc_run_cpu_sequencer_v17_00):].cast('I')[:hdr.cmdIndex])

    for op in cmd_iter:
      if op == 0x0: self.nvdev.wreg(next(cmd_iter), next(cmd_iter)) # reg write
      elif op == 0x1: # reg modify
        addr, val, mask = next(cmd_iter), next(cmd_iter), next(cmd_iter)
        self.nvdev.wreg(addr, (self.nvdev.rreg(addr) & ~mask) | (val & mask))
      elif op == 0x2: # reg poll
        addr, mask, val, _, _ = next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter), next(cmd_iter)
        while (self.nvdev.rreg(addr) & mask) != val: pass
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

        self.nvdev.NV_PGSP_FALCON_MAILBOX0.write(lo32(self.libos_args_sysmem[0]))
        self.nvdev.NV_PGSP_FALCON_MAILBOX1.write(hi32(self.libos_args_sysmem[0]))

        self.nvdev.flcn.start_cpu(self.nvdev.flcn.sec2)
        while self.nvdev.NV_PGC6_BSI_SECURE_SCRATCH_14.read_bitfields()['boot_stage_3_handoff'] == 0: pass

        mailbox = self.nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base(self.nvdev.flcn.sec2).read()
        assert mailbox == 0x0, f"Falcon SEC2 failed to execute, mailbox is {mailbox:08x}"
      else: raise ValueError(f"Unknown op code {op} in run_cpu_seq")
