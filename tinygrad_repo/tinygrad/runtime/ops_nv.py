from __future__ import annotations
import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, weakref
assert sys.platform != 'win32'
from typing import cast, ClassVar
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQProgram, HCQSignal, BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface, FileIOInterface, MOCKGPU, hcq_filter_visible_devices, hcq_profile
from tinygrad.uop.ops import sint
from tinygrad.device import Compiled, BufferSpec, CompilerPair, CompilerSet
from tinygrad.helpers import getenv, mv_address, round_up, data64, data64_le, prod, OSX, to_mv, hi32, lo32, NV_CC, NV_PTX, NV_NAK, PROFILE
from tinygrad.helpers import ContextVar, VIZ, ProfileEvent
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import NVRenderer
from tinygrad.runtime.support.compiler_cuda import CUDACompiler, PTXCompiler, NVPTXCompiler, NVCompiler
from tinygrad.runtime.autogen import nv_570, nv_580, pci, mesa
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.system import System, PCIIfaceBase, MAP_FIXED
from tinygrad.renderer.nir import NAKRenderer
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401 # pylint: disable=unused-import

nv_gpu = nv_570 # default to 570

PMA = ContextVar("PMA", abs(VIZ.value)>=2)

@dataclass(frozen=True)
class ProfilePMAEvent(ProfileEvent): device:str; kern:str; blob:bytes # noqa: E702

class NVSignal(HCQSignal):
  def _sleep(self, time_spent_waiting_ms:int) -> bool:
    # Resonable to sleep for long workloads (which take more than 2s) and only timeline signals.
    if time_spent_waiting_ms > 2000 and self.is_timeline and self.owner is not None: return self.owner.iface.sleep(200)
    return False

def get_error_str(status): return f"{status}: {nv_gpu.nv_status_codes.get(status, 'Unknown error')}"

NV_PFAULT_FAULT_TYPE = {dt:name for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_FAULT_TYPE_")}
NV_PFAULT_ACCESS_TYPE = {dt:name.split("_")[-1] for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_ACCESS_TYPE_")}

def nv_iowr(fd:FileIOInterface, nr, args, cmd=None):
  ret = fd.ioctl(cmd or ((3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF)), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

class QMD:
  fields: dict[str, dict[str, tuple[int, int]]] = {}

  def __init__(self, dev:NVDevice, addr:int|None=None, **kwargs):
    self.ver, self.sz = (5, 0x60) if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)

    # Init fields from module
    if (pref:="NVCEC0_QMDV05_00" if self.ver == 5 else "NVC6C0_QMDV03_00") not in QMD.fields:
      QMD.fields[pref] = {**{name[len(pref)+1:]: dt for name,dt in nv_gpu.__dict__.items() if name.startswith(pref) and isinstance(dt, tuple)},
        **{name[len(pref)+1:]+f"_{i}": dt(i) for name,dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith(pref) and callable(dt)}}

    self.mv, self.pref = (memoryview(bytearray(self.sz * 4)) if addr is None else to_mv(addr, self.sz * 4)), pref
    if kwargs: self.write(**kwargs)

  def _rw_bits(self, hi:int, lo:int, value:int|None=None):
    mask = ((1 << (width:=hi - lo + 1)) - 1) << (lo % 8)
    num = int.from_bytes(self.mv[lo//8:hi//8+1], "little")

    if value is None: return (num & mask) >> (lo % 8)

    if value >= (1 << width): raise ValueError(f"{value:#x} does not fit.")
    self.mv[lo//8:hi//8+1] = int((num & ~mask) | ((value << (lo % 8)) & mask)).to_bytes((hi//8 - lo//8 + 1), "little")

  def write(self, **kwargs):
    for k,val in kwargs.items(): self._rw_bits(*QMD.fields[self.pref][k.upper()], value=val) # type: ignore [misc]

  def read(self, k, val=0): return self._rw_bits(*QMD.fields[self.pref][k.upper()])

  def field_offset(self, k): return QMD.fields[self.pref][k.upper()][1] // 8

  def set_constant_buf_addr(self, i, addr):
    if self.ver < 4: self.write(**{f'constant_buffer_addr_upper_{i}':hi32(addr), f'constant_buffer_addr_lower_{i}':lo32(addr)})
    else: self.write(**{f'constant_buffer_addr_upper_shifted6_{i}':hi32(addr >> 6), f'constant_buffer_addr_lower_shifted6_{i}':lo32(addr >> 6)})

class NVCommandQueue(HWQueue[HCQSignal, 'NVDevice', 'NVProgram', 'NVArgsState']):
  def __init__(self):
    self.active_qmd = None
    super().__init__()

  def __del__(self):
    if self.binded_device is not None: self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True))

  def nvm(self, subchannel, mthd, *args, typ=2): self.q((typ << 28) | (len(args) << 16) | (subchannel << 13) | (mthd >> 2), *args)

  def setup(self, compute_class=None, copy_class=None, local_mem_window=None, shared_mem_window=None, local_mem=None, local_mem_tpc_bytes=None):
    if compute_class: self.nvm(1, nv_gpu.NVC6C0_SET_OBJECT, compute_class)
    if copy_class: self.nvm(4, nv_gpu.NVC6C0_SET_OBJECT, copy_class)
    if local_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(local_mem_window))
    if shared_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(shared_mem_window))
    if local_mem: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(local_mem))
    if local_mem_tpc_bytes: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(local_mem_tpc_bytes), 0xff)
    return self

  def wait(self, signal:HCQSignal, value:sint=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value), (3 << 0) | (1 << 24)) # ACQUIRE | PAYLOAD_SIZE_64BIT
    self.active_qmd = None
    return self

  def timestamp(self, signal:HCQSignal): return self.signal(signal, 0)

  def bind(self, dev:NVDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True))
    hw_view = self.hw_page.cpu_view().view(fmt='I')
    for i, value in enumerate(self._q): hw_view[i] = value

    # From now on, the queue is on the device for faster submission.
    self._q = hw_view

  def _submit_to_gpfifo(self, dev:NVDevice, gpfifo:GPFifo):
    if dev == self.binded_device: cmdq_addr = self.hw_page.va_addr
    else:
      cmdq_addr = dev.cmdq_allocator.alloc(len(self._q) * 4, 16)
      cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
      dev.cmdq[cmdq_wptr : cmdq_wptr + len(self._q)] = array.array('I', self._q)

    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr//4 << 2) | (len(self._q) << 42) | (1 << 41)
    gpfifo.gpput[0] = (gpfifo.put_value + 1) % gpfifo.entries_count

    System.memory_barrier()
    dev.gpu_mmio[0x90 // 4] = gpfifo.token
    gpfifo.put_value += 1

class NVComputeQueue(NVCommandQueue):
  def memory_barrier(self):
    self.nvm(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI, (1 << 12) | (1 << 4) | (1 << 0))
    self.active_qmd:QMD|None = None
    return self

  def exec(self, prg:NVProgram, args_state:NVArgsState, global_size:tuple[sint, ...], local_size:tuple[sint, ...]):
    self.bind_args_state(args_state)

    qmd_buf = args_state.buf.offset(round_up(prg.constbufs[0][1], 1 << 8))
    qmd_buf.cpu_view().view(size=prg.qmd.mv.nbytes, fmt='B')[:] = prg.qmd.mv
    assert qmd_buf.va_addr < (1 << 40), f"large qmd addr {qmd_buf.va_addr:x}"

    qmd = QMD(dev=prg.dev, addr=qmd_buf.cpu_view().addr) # Save qmd for later update

    self.bind_sints_to_mem(*global_size, mem=qmd_buf.cpu_view(), fmt='I', offset=qmd.field_offset('cta_raster_width' if qmd.ver<4 else 'grid_width'))
    self.bind_sints_to_mem(*(local_size[:2]), mem=qmd_buf.cpu_view(), fmt='H', offset=qmd.field_offset('cta_thread_dimension0'))
    self.bind_sints_to_mem(local_size[2], mem=qmd_buf.cpu_view(), fmt='B', offset=qmd.field_offset('cta_thread_dimension2'))
    qmd.set_constant_buf_addr(0, args_state.buf.va_addr)

    if self.active_qmd is None:
      if prg.dev.pma_enabled: self.nvm(1, nv_gpu.NVC6C0_PM_TRIGGER, 0)
      self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, qmd_buf.va_addr >> 8)
      self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, 9)
    else:
      self.active_qmd.write(dependent_qmd0_pointer=qmd_buf.va_addr >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)

    self.active_qmd, self.active_qmd_buf = qmd, qmd_buf
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    if self.active_qmd is not None:
      for i in range(2):
        if self.active_qmd.read(f'release{i}_enable') == 0:
          self.active_qmd.write(**{f'release{i}_enable': 1})

          addr_off = self.active_qmd.field_offset(f'release{i}_address_lower' if self.active_qmd.ver<4 else f'release_semaphore{i}_addr_lower')
          self.bind_sints_to_mem(signal.value_addr & 0xffffffff, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=addr_off)
          self.bind_sints_to_mem(signal.value_addr >> 32, mem=self.active_qmd_buf.cpu_view(), fmt='I', mask=0xf, offset=addr_off+4)

          val_off = self.active_qmd.field_offset(f'release{i}_payload_lower' if self.active_qmd.ver<4 else f'release_semaphore{i}_payload_lower')
          self.bind_sints_to_mem(value & 0xffffffff, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=val_off)
          self.bind_sints_to_mem(value >> 32, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=val_off+4)
          return self

    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             (1 << 0) | (1 << 20) | (1 << 24) | (1 << 25)) # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
    self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)
    self.active_qmd = None
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.compute_gpfifo)

class NVCopyQueue(NVCommandQueue):
  def __init__(self, queue_idx=0):
    self.queue_idx = queue_idx
    super().__init__()

  def copy(self, dest:sint, src:sint, copy_size:int):
    for off in range(0, copy_size, step:=(1 << 31)):
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *data64(src+off), *data64(dest+off))
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, min(copy_size-off, step))
      self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x182) # TRANSFER_TYPE_NON_PIPELINED | DST_MEMORY_LAYOUT_PITCH | SRC_MEMORY_LAYOUT_PITCH
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x14)
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.dma_gpfifo)

class NVVideoQueue(NVCommandQueue):
  def decode_hevc_chunk(self, pic_desc:HCQBuffer, in_buf:HCQBuffer, out_buf:HCQBuffer, out_buf_pos:int, hist_bufs:list[HCQBuffer], hist_pos:list[int],
                        chroma_off:int, coloc_buf:HCQBuffer, filter_buf:HCQBuffer, intra_top_off:int, intra_unk_off:int|None, status_buf:HCQBuffer):
    self.nvm(4, nv_gpu.NVC9B0_SET_APPLICATION_ID, nv_gpu.NVC9B0_SET_APPLICATION_ID_ID_HEVC)
    self.nvm(4, nv_gpu.NVC9B0_SET_CONTROL_PARAMS, 0x52057)
    self.nvm(4, nv_gpu.NVC9B0_SET_DRV_PIC_SETUP_OFFSET, pic_desc.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_IN_BUF_BASE_OFFSET, in_buf.va_addr >> 8)
    for pos, buf in zip(hist_pos + [out_buf_pos], hist_bufs + [out_buf]):
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_LUMA_OFFSET0 + pos*4, buf.va_addr >> 8)
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_CHROMA_OFFSET0 + pos*4, buf.offset(chroma_off).va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_COLOC_DATA_OFFSET, coloc_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_NVDEC_STATUS_OFFSET, status_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_TILE_SIZES_OFFSET, pic_desc.offset(0x200).va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_FILTER_BUFFER_OFFSET, filter_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_INTRA_TOP_BUF_OFFSET, (filter_buf.va_addr + intra_top_off) >> 8)
    if intra_unk_off is not None: self.nvm(4, 0x4dc, (filter_buf.va_addr + intra_unk_off) >> 8)
    self.nvm(4, nv_gpu.NVC9B0_EXECUTE, 0)
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC9B0_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC9B0_SEMAPHORE_D, (1 << 24) | (1 << 0))
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.vid_gpfifo)

class NVArgsState(CLikeArgsState):
  def __init__(self, buf:HCQBuffer, prg:NVProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    if MOCKGPU: prg.cbuf_0[80:82] = [len(bufs), len(vals)]
    super().__init__(buf, prg, bufs, vals=vals, prefix=prg.cbuf_0 or None)

class NVProgram(HCQProgram):
  def __init__(self, dev:NVDevice, name:str, lib:bytes, **kwargs):
    self.dev, self.name, self.lib = dev, name, lib
    self.constbufs: dict[int, tuple[int, int]] = {0: (0, 0x160)} # dict[constbuf index, tuple[va_addr, size]]

    if (NAK:=isinstance(dev.renderer, NAKRenderer)):
      image, self.cbuf_0 = memoryview(bytearray(lib[ctypes.sizeof(info:=mesa.struct_nak_shader_info.from_buffer_copy(lib)):])), []
      self.regs_usage, self.shmem_usage, self.lcmem_usage = info.num_gprs, round_up(info.cs.smem_size, 128), round_up(info.slm_size, 16)
    elif MOCKGPU: image, sections, relocs = memoryview(bytearray(lib) + b'\x00' * (4 - len(lib)%4)).cast("I"), [], [] # type: ignore
    else: image, sections, relocs = elf_loader(self.lib, force_section_align=128)
    # NOTE: Ensure at least 4KB of space after the program to mitigate prefetch memory faults.
    self.lib_gpu = self.dev.allocator.alloc(round_up((prog_sz:=image.nbytes), 0x1000) + 0x1000, buf_spec:=BufferSpec(nolru=True))
    prog_addr = self.lib_gpu.va_addr
    if not NAK:
      # For MOCKGPU, the lib is PTX code, so some values are emulated.
      self.regs_usage, self.shmem_usage, self.lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0 if not MOCKGPU else 0x160
      for sh in sections: # pylint: disable=possibly-used-before-assignment
        if sh.name == f".nv.shared.{self.name}": self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
        if sh.name == f".text.{self.name}": prog_addr, prog_sz = self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size
        elif m:=re.match(r'\.nv\.constant(\d+)', sh.name):
          self.constbufs[int(m.group(1))] = (self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size)
        elif sh.name.startswith(".nv.info"):
          for typ, param, data in self._parse_elf_info(sh):
            if sh.name == f".nv.info.{name}" and param == 0xa: cbuf0_size = struct.unpack_from("IH", data)[1] # EIATTR_PARAM_CBANK
            elif sh.name == ".nv.info" and param == 0x12: self.lcmem_usage = struct.unpack_from("II", data)[1] + 0x240 # EIATTR_MIN_STACK_SIZE
            elif sh.name == ".nv.info" and param == 0x2f: self.regs_usage = struct.unpack_from("II", data)[1] # EIATTR_REGCOUNT

      # Apply relocs
      for apply_image_offset, rel_sym_offset, typ, _ in relocs: # pylint: disable=possibly-used-before-assignment
        # These types are CUDA-specific, applying them here
        if typ == 2: image[apply_image_offset:apply_image_offset+8] = struct.pack('<Q', self.lib_gpu.va_addr + rel_sym_offset) # R_CUDA_64
        elif typ == 0x38: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) & 0xffffffff)
        elif typ == 0x39: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
        else: raise RuntimeError(f"unknown NV reloc {typ}")

      # Minimum cbuf_0 size for driver params: Blackwell needs index 223 (224 entries), older GPUs need index 11 (12 entries)
      min_cbuf0_entries = 224 if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 12
      self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)

    # Ensure device has enough local memory to run the program
    self.dev._ensure_has_local_memory(self.lcmem_usage)
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()

    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      if not NAK: self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xfffdc0
      qmd = {'qmd_major_version':5, 'qmd_type':nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, 'program_address_upper_shifted4':hi32(prog_addr>>4),
        'program_address_lower_shifted4':lo32(prog_addr>>4), 'register_count':self.regs_usage, 'shared_memory_size_shifted7':self.shmem_usage>>7,
        'shader_local_memory_high_size_shifted4':self.lcmem_usage>>4 if NAK else self.dev.slm_per_thread>>4}
    else:
      if not NAK: self.cbuf_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xfffdc0)]
      qmd = {'qmd_major_version':3, 'sm_global_caching_enable':1, 'program_address_upper':hi32(prog_addr), 'program_address_lower':lo32(prog_addr),
        'shared_memory_size':self.shmem_usage, 'register_count_v':self.regs_usage,
        **({'shader_local_memory_low_size':self.lcmem_usage} if NAK else {'shader_local_memory_high_size':self.dev.slm_per_thread})}

    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1

    self.qmd:QMD = QMD(dev, **qmd, qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalidate_0=1, min_sm_config_shared_mem_size=smem_cfg,
      target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a, program_prefetch_size=min(prog_sz>>8, 0x1ff),
      sass_version=dev.sass_version, program_prefetch_addr_upper_shifted=prog_addr>>40, program_prefetch_addr_lower_shifted=prog_addr>>8)

    for i,(addr,sz) in self.constbufs.items():
      self.qmd.set_constant_buf_addr(i, addr)
      self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})

    # Registers allocation granularity per warp is 256, warp allocation granularity is 4. Register file size is 65536.
    self.max_threads = ((65536 // round_up(max(1, self.regs_usage) * 32, 256)) // 4) * 4 * 32

    # NV's kernargs is constbuffer, then arguments to the kernel follows. Kernargs also appends QMD at the end of the kernel.
    super().__init__(NVArgsState, self.dev, self.name, kernargs_alloc_size=round_up(self.constbufs[0][1], 1 << 8) + (8 << 8))
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off+4:start_off+sz+4] if typ == 0x4 else sz
      start_off += (sz if typ == 0x4 else 0) + 4

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int|None, ...]=(), wait=False):
    if prod(local_size) > 1024 or self.max_threads < prod(local_size) or self.lcmem_usage > cast(NVDevice, self.dev).slm_per_thread:
      raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {self.max_threads=}")
    if any(cur > mx for cur,mx in zip(global_size, [2147483647, 65535, 65535])) or any(cur > mx for cur,mx in zip(local_size, [1024, 1024, 64])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    res = super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)
    if self.dev.pma_enabled:
      self.dev.synchronize()
      if pma_blob:=self.dev._prof_readback(): Compiled.profile_events += [ProfilePMAEvent(self.dev.device, self.name, pma_blob)]
    return res

class NVAllocator(HCQAllocator['NVDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev.iface.alloc(size, cpu_access=options.cpu_access, host=options.host)

  def _do_free(self, opaque:HCQBuffer, options:BufferSpec): self.dev.iface.free(opaque)

  def _map(self, buf:HCQBuffer): return self.dev.iface.map(buf._base if buf._base is not None else buf)

  def _encode_decode(self, bufout:HCQBuffer, bufin:HCQBuffer, desc_buf:HCQBuffer, hist:list[HCQBuffer], shape:tuple[int,...], frame_pos:int):
    assert all(h.va_addr % 0x100 == 0 for h in hist + [bufin, bufout, desc_buf]), "all buffers must be 0x100 aligned"

    h, w = ((2 * shape[0]) // 3 if shape[0] % 3 == 0 else (2 * shape[0] - 1) // 3), shape[1]
    self.dev._ensure_has_vid_hw(w, h)

    q = NVVideoQueue().wait(self.dev.timeline_signal, self.dev.timeline_value - 1)
    with hcq_profile(self.dev, queue=q, desc="NVDEC", enabled=PROFILE):
      q.decode_hevc_chunk(desc_buf, bufin, bufout, frame_pos, hist, [(frame_pos-x) % (len(hist) + 1) for x in range(len(hist), 0, -1)],
                          round_up(w, 64)*round_up(h, 64), self.dev.vid_coloc_buf, self.dev.vid_filter_buf, self.dev.intra_top_off,
                          self.dev.intra_unk_off, self.dev.vid_stat_buf)
    q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)

@dataclass
class GPFifo:
  ring: MMIOInterface
  gpput: MMIOInterface
  entries_count: int
  token: int
  put_value: int = 0

class NVKIface:
  root = None
  fd_ctl: FileIOInterface
  fd_uvm: FileIOInterface
  gpus_info: list|ctypes.Array = []

  # TODO: Need a proper allocator for va addresses
  # 0x1000000000 - 0x2000000000, reserved for system/cpu mappings
  # VA space is 48bits.
  low_uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=0x1000000000, base=0x8000000000 if OSX else 0x1000000000, wrap=False)
  uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=(1 << 48) - 1, base=low_uvm_vaddr_allocator.base + low_uvm_vaddr_allocator.size, wrap=False)
  host_object_enumerator: int = 0x1000

  def __init__(self, dev, device_id):
    if NVKIface.root is None:
      global nv_gpu

      NVKIface.fd_ctl = FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.fd_uvm = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      self.fd_uvm_2 = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.root = self.rm_alloc(0, nv_gpu.NV01_ROOT_CLIENT, None, root=0)

      drvver = self.rm_control(self.root, nv_gpu.NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2, nv_gpu.NV0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS())
      if int(drvver.driverVersionBuffer.decode().split('.')[0], 10) >= 580: nv_gpu = nv_580

      self.uvm(nv_gpu.UVM_INITIALIZE, nv_gpu.UVM_INITIALIZE_PARAMS())

      # this error is okay, CUDA hits it too
      with contextlib.suppress(RuntimeError): self.uvm(nv_gpu.UVM_MM_INITIALIZE, nv_gpu.UVM_MM_INITIALIZE_PARAMS(uvmFd=self.fd_uvm.fd), self.fd_uvm_2)

      nv_iowr(NVKIface.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, gpus_info:=(nv_gpu.nv_ioctl_card_info_t*64)())
      NVKIface.gpus_info = hcq_filter_visible_devices(gpus_info)

    self.dev, self.device_id = dev, device_id
    if self.device_id >= len(NVKIface.gpus_info) or not NVKIface.gpus_info[self.device_id].valid:
      raise RuntimeError(f"No device found for {device_id}. Requesting more devices than the system has?")

    self.fd_dev = self._new_gpu_fd()
    self.gpu_info = self.rm_control(self.root, nv_gpu.NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2,
      nv_gpu.NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS(gpuId=NVKIface.gpus_info[self.device_id].gpu_id))
    self.gpu_minor = NVKIface.gpus_info[self.device_id].minor_number
    self.gpu_instance = self.gpu_info.deviceInstance

  def rm_alloc(self, parent, clss, params=None, root=None) -> int:
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_ALLOC, made:=nv_gpu.NVOS21_PARAMETERS(hRoot=root if root is not None else self.root,
      hObjectParent=parent, hClass=clss, pAllocParms=ctypes.cast(ctypes.byref(params), ctypes.c_void_p) if params is not None else None))
    if made.status == nv_gpu.NV_ERR_NO_MEMORY: raise MemoryError(f"rm_alloc returned {get_error_str(made.status)}")
    if made.status != 0: raise RuntimeError(f"rm_alloc returned {get_error_str(made.status)}")
    return made.hObjectNew

  def rm_control(self, obj, cmd, params=None):
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_CONTROL, made:=nv_gpu.NVOS54_PARAMETERS(hClient=self.root, hObject=obj, cmd=cmd,
      paramsSize=ctypes.sizeof(params) if params is not None else 0,
      params=ctypes.cast(ctypes.byref(params), ctypes.c_void_p) if params is not None else None))
    if made.status != 0: raise RuntimeError(f"rm_control returned {get_error_str(made.status)}")
    return params

  def uvm(self, cmd, params, fd=None):
    nv_iowr(fd or self.fd_uvm, None, params, cmd=cmd)
    if params.rmStatus != 0: raise RuntimeError(f"uvm returned {get_error_str(params.rmStatus)}")

  def setup_usermode(self):
    clsnum = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS(numClasses=0))
    clsinfo = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS(
      numClasses=clsnum.numClasses, classList=mv_address(classlist:=memoryview(bytearray(clsnum.numClasses * 4)).cast('I'))))
    self.nvclasses = {classlist[i] for i in range(clsinfo.numClasses)}
    self.usermode_class:int = next(c for c in [nv_gpu.HOPPER_USERMODE_A, nv_gpu.TURING_USERMODE_A] if c in self.nvclasses)
    self.gpfifo_class:int = next(c for c in [nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_CHANNEL_GPFIFO_A] if c in self.nvclasses)
    self.compute_class:int = next(c for c in [nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.ADA_COMPUTE_A, nv_gpu.AMPERE_COMPUTE_B] if c in self.nvclasses)
    self.dma_class:int = next(c for c in [nv_gpu.BLACKWELL_DMA_COPY_B, nv_gpu.AMPERE_DMA_COPY_B] if c in self.nvclasses)
    self.viddec_class:int|None = next((c for c in [nv_gpu.NVCFB0_VIDEO_DECODER, nv_gpu.NVC9B0_VIDEO_DECODER] if c in self.nvclasses), None)

    usermode = self.rm_alloc(self.dev.subdevice, self.usermode_class)
    return usermode, MMIOInterface(self._gpu_map_to_cpu(usermode, mmio_sz:=0x10000), mmio_sz, fmt='I')

  def setup_vm(self, vaspace):
    self.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO, raw_uuid:=nv_gpu.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(
      flags=nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16))
    self.gpu_uuid = nv_gpu.struct_nv_uuid(uuid=(ctypes.c_ubyte*16)(*[raw_uuid.data[i] for i in range(16)]))

    self.uvm(nv_gpu.UVM_REGISTER_GPU, nv_gpu.UVM_REGISTER_GPU_PARAMS(rmCtrlFd=-1, gpu_uuid=self.gpu_uuid))
    self.uvm(nv_gpu.UVM_REGISTER_GPU_VASPACE, nv_gpu.UVM_REGISTER_GPU_VASPACE_PARAMS(
      gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hVaSpace=vaspace))

    for dev in cast(list[NVDevice], [d for pg in HCQCompiled.peer_groups.values() for d in pg if isinstance(d, NVDevice) and not d.is_nvd()]):
      try: self.uvm(nv_gpu.UVM_ENABLE_PEER_ACCESS, nv_gpu.UVM_ENABLE_PEER_ACCESS_PARAMS(gpuUuidA=self.gpu_uuid, gpuUuidB=dev.iface.gpu_uuid))
      except RuntimeError as e: raise RuntimeError(f"{e}. Make sure GPUs #{self.gpu_minor} & #{dev.iface.gpu_minor} have P2P enabled.") from e

  def setup_gpfifo_vm(self, gpfifo):
    self.uvm(nv_gpu.UVM_REGISTER_CHANNEL, nv_gpu.UVM_REGISTER_CHANNEL_PARAMS(gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root,
      hChannel=gpfifo, base=self._alloc_gpu_vaddr(0x4000000, force_low=True), length=0x4000000))

  def _new_gpu_fd(self):
    fd_dev = FileIOInterface(f"/dev/nvidia{NVKIface.gpus_info[self.device_id].minor_number}", os.O_RDWR | os.O_CLOEXEC)
    nv_iowr(fd_dev, nv_gpu.NV_ESC_REGISTER_FD, nv_gpu.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl.fd))
    return fd_dev

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0, system=False):
    fd_dev = self._new_gpu_fd() if not system else FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev.fd,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.dev.nvdevice, hMemory=memory_handle, length=size, flags=flags))
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_gpu_map_to_cpu returned {get_error_str(made.params.status)}")
    return fd_dev.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), 0)

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, map_flags=0, cpu_addr=None, **kwargs) -> HCQBuffer:
    # Uncached memory is "system". Use huge pages only for gpu memory.
    page_size = mmap.PAGESIZE if uncached or host else ((2 << 20) if size >= (8 << 20) else (mmap.PAGESIZE if MOCKGPU else 4 << 10))
    size = round_up(size, page_size)
    va_addr = self._alloc_gpu_vaddr(size, alignment=page_size, force_low=cpu_access) if (alloced:=cpu_addr is None) else cpu_addr

    if host:
      if alloced: va_addr = FileIOInterface.anon_mmap(va_addr, size, mmap.PROT_READ|mmap.PROT_WRITE, MAP_FIXED|mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, 0)

      flags = (nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (nv_gpu.NVOS02_FLAGS_COHERENCY_CACHED << 12) \
            | (nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30)

      NVKIface.host_object_enumerator += 1
      made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.dev.nvdevice, flags=flags,
        hObjectNew=NVKIface.host_object_enumerator, hClass=nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, pMemory=va_addr, limit=size-1), fd=-1)
      nv_iowr(self.fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)

      if made.params.status != 0: raise RuntimeError(f"host alloc returned {get_error_str(made.params.status)}")
      mem_handle = made.params.hObjectNew
    else:
      attr = ((nv_gpu.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS if contiguous else nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) << 27) \
          | (nv_gpu.NVOS32_ATTR_PAGE_SIZE_HUGE if page_size > 0x1000 else 0) << 23 | ((nv_gpu.NVOS32_ATTR_LOCATION_PCI if uncached else 0) << 25)

      attr2 = ((nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_NO if uncached else nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_YES) << 2) \
            | ((nv_gpu.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB if page_size > 0x1000 else 0) << 20) | nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC \
            | ((nv_gpu.NVOS32_ATTR2_PROTECTION_USER_READ_ONLY << 22) if kwargs.get('read_only') else 0)

      fl = nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | nv_gpu.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE \
         | nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | (nv_gpu.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM if not uncached else 0)

      alloc_func = nv_gpu.NV1_MEMORY_SYSTEM if uncached else nv_gpu.NV1_MEMORY_USER
      alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, alignment=page_size, offset=0, limit=size-1, format=6, size=size,
        type=nv_gpu.NVOS32_TYPE_NOTIFIER if uncached else nv_gpu.NVOS32_TYPE_IMAGE, attr=attr, attr2=attr2, flags=fl)
      mem_handle = self.rm_alloc(self.dev.nvdevice, alloc_func, alloc_params)

      if cpu_access: va_addr = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags, system=uncached)

    return self._gpu_uvm_map(va_addr, size, mem_handle, has_cpu_mapping=cpu_access or host)

  def free(self, mem:HCQBuffer):
    if mem.meta.hMemory > NVKIface.host_object_enumerator: # not a host object, clear phys mem.
      made = nv_gpu.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.dev.nvdevice, hObjectOld=mem.meta.hMemory)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_FREE, made)
      if made.status != 0: raise RuntimeError(f"_gpu_free returned {get_error_str(made.status)}")

    self.uvm(nv_gpu.UVM_FREE, nv_gpu.UVM_FREE_PARAMS(base=int(mem.va_addr), length=mem.size))
    if mem.view is not None: FileIOInterface.munmap(int(mem.va_addr), mem.size)

  def _gpu_uvm_map(self, va_base, size, mem_handle, create_range=True, has_cpu_mapping=False) -> HCQBuffer:
    if create_range:
      self.uvm(nv_gpu.UVM_CREATE_EXTERNAL_RANGE, nv_gpu.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=va_base, length=size))
      made = nv_gpu.NVOS46_PARAMETERS(hClient=self.root, hDevice=self.dev.nvdevice, hDma=self.dev.virtmem, hMemory=mem_handle, length=size,
        flags=(nv_gpu.NVOS46_FLAGS_PAGE_SIZE_4KB<<8)|(nv_gpu.NVOS46_FLAGS_CACHE_SNOOP_ENABLE<<4)|(nv_gpu.NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE<<15),
        dmaOffset=va_base)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY_DMA, made)
      if made.status != 0: raise RuntimeError(f"nv_sys_alloc 1 returned {get_error_str(made.status)}")
      assert made.dmaOffset == va_base, f"made.dmaOffset != va_base {made.dmaOffset=} {va_base=}"

    attrs = (nv_gpu.UvmGpuMappingAttributes*256)(nv_gpu.UvmGpuMappingAttributes(gpuUuid=self.gpu_uuid, gpuMappingType=1))

    self.uvm(nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION, uvm_map:=nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=va_base, length=size,
      rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hMemory=mem_handle, gpuAttributesCount=1, perGpuAttributes=attrs, mapped_gpu_ids=[self.gpu_uuid]))
    return HCQBuffer(va_base, size, meta=uvm_map, view=MMIOInterface(va_base, size, fmt='B') if has_cpu_mapping else None, owner=self.dev)

  def map(self, mem:HCQBuffer):
    if mem.owner is not None and mem.owner._is_cpu():
      if not any(x.device.startswith("NV") for x in mem.mapped_devs): return self.alloc(mem.size, host=True, cpu_addr=mem.va_addr)
      mem = mem.mappings[next(x for x in mem.mapped_devs if x.device.startswith("NV"))]
    self._gpu_uvm_map(mem.va_addr, mem.size, mem.meta.hMemory, create_range=False)

  def _alloc_gpu_vaddr(self, size, alignment=(4 << 10), force_low=False):
    return NVKIface.low_uvm_vaddr_allocator.alloc(size, alignment) if force_low else NVKIface.uvm_vaddr_allocator.alloc(size, alignment)

  def sleep(self, tm:int) -> bool: return False

class PCIIface(PCIIfaceBase):
  gpus:ClassVar[list[str]] = []

  def __init__(self, dev, dev_id):
    # PCIIface's MAP_FIXED mmap will overwrite UVM allocations made by NVKIface, so don't try PCIIface if kernel driver was already used.
    if NVKIface.root is not None: raise RuntimeError("Cannot use PCIIface after NVKIface has been initialized (would corrupt UVM memory)")
    super().__init__(dev, dev_id, vendor=0x10de, devices=[(0xff00, [0x2200, 0x2400, 0x2500, 0x2600, 0x2700, 0x2800, 0x2b00, 0x2c00, 0x2d00, 0x2f00])],
      base_class=0x03, bars=[0, 1, 3], vram_bar=1, va_start=NVMemoryManager.va_allocator.base, va_size=NVMemoryManager.va_allocator.size)
    if not OSX: System.reserve_hugepages(64)

    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.dev_impl:NVDev = NVDev(self.pci_dev)
    self.root, self.gpu_instance = 0xc1000000, 0
    self.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())

    # Setup classes for the GPU
    self.gpfifo_class, self.compute_class, self.dma_class = (gsp:=self.dev_impl.gsp).gpfifo_class, gsp.compute_class, gsp.dma_class
    self.viddec_class = None

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, **kwargs) -> HCQBuffer:
    # Force use of huge pages for large allocations. NVDev will attempt to use huge pages in any case,
    # but if the size is not aligned, the tail will be allocated with 4KB pages, increasing TLB pressure.
    page_size = mmap.PAGESIZE if uncached or host else ((2 << 20) if size >= (8 << 20) else (4 << 10))
    return super().alloc(round_up(size, page_size), host=host, uncached=uncached, cpu_access=cpu_access, contiguous=contiguous, **kwargs)

  def setup_usermode(self): return 0xce000000, self.pci_dev.map_bar(bar=0, fmt='I', off=0xbb0000, size=0x10000)
  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass

  def rm_alloc(self, parent, clss, params=None, root=None) -> int: return self.dev_impl.gsp.rpc_rm_alloc(parent, clss, params, self.root)
  def rm_control(self, obj, cmd, params=None): return self.dev_impl.gsp.rpc_rm_control(obj, cmd, params, self.root)

  def device_fini(self): self.dev_impl.fini()

  def sleep(self, timeout) -> bool:
    for _ in self.dev_impl.gsp.stat_q.read_resp(): pass
    return self.dev_impl.is_err_state

class NVDevice(HCQCompiled[NVSignal]):
  def is_nvd(self) -> bool: return isinstance(self.iface, PCIIface)

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = self._select_iface(NVKIface, PCIIface)

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root,
                                                   vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES)
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())
    self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL, nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(limit=0x1ffffffffffff))
    self.usermode, self.gpu_mmio = self.iface.setup_usermode()

    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff,
      flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) | \
             (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX))))

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A, vaspace_params)

    self.iface.setup_vm(vaspace)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    self.channel_group = self.iface.rm_alloc(self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, channel_params)

    self.gpfifo_area = self.iface.alloc(0x300000, contiguous=True, cpu_access=True, force_devmem=True,
      map_flags=(nv_gpu.NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED<<23))

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = self.iface.rm_alloc(self.channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A, ctxshare_params)

    self.compute_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0, entries=0x10000, compute=True)
    self.dma_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0x100000, entries=0x10000, compute=False)
    self.iface.rm_control(self.channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))

    self.cmdq_page:HCQBuffer = self.iface.alloc(0x200000, cpu_access=True)
    self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=int(self.cmdq_page.va_addr), wrap=True)
    self.cmdq = self.cmdq_page.cpu_view().view(fmt='I')

    self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version = self._query_gpu_info('num_gpcs',
      'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version')

    # FIXME: no idea how to convert this for blackwells
    self.arch: str = "sm_120" if self.sm_version==0xa04 else f"sm_{(self.sm_version>>8)&0xff}{(val>>4) if (val:=self.sm_version&0xff) > 0xf else val}"
    self.sass_version = ((self.sm_version & 0xf00) >> 4) | (self.sm_version & 0xf)

    cucc, ptxcc = (CUDACompiler, PTXCompiler) if MOCKGPU else (NVCompiler, NVPTXCompiler)
    compilers = CompilerSet(ctrl_var=NV_CC, cset=[CompilerPair(functools.partial(NVRenderer, self.arch),functools.partial(cucc, self.arch)),
       CompilerPair(functools.partial(PTXRenderer, self.arch, device="NV"), functools.partial(ptxcc, self.arch), NV_PTX),
       CompilerPair(functools.partial(NAKRenderer, self.arch, self.max_warps_per_sm), None, NV_NAK)])
    super().__init__(device, NVAllocator(self), compilers, functools.partial(NVProgram, self), NVSignal, NVComputeQueue, NVCopyQueue)

    self.pma_enabled = PMA.value > 0 and PROFILE >= 1
    if self.pma_enabled: self._prof_init()

    self._setup_gpfifos()

  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, compute=False, video=False) -> GPFifo:
    notifier = self.iface.alloc(48 << 20, uncached=True)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr+offset, gpFifoEntries=entries, hContextShare=ctxshare,
      hObjectError=notifier.meta.hMemory, hObjectBuffer=self.virtmem if video else gpfifo_area.meta.hMemory,
      hUserdMemory=(ctypes.c_uint32*8)(gpfifo_area.meta.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset), engineType=19 if video else 0)
    gpfifo = self.iface.rm_alloc(channel_group, self.iface.gpfifo_class, params)

    if compute:
      self.debug_compute_obj, self.debug_channel = self.iface.rm_alloc(gpfifo, self.iface.compute_class), gpfifo
      debugger_params = nv_gpu.NV83DE_ALLOC_PARAMETERS(hAppClient=self.iface.root, hClass3dObject=self.debug_compute_obj)
      self.debugger = self.iface.rm_alloc(self.nvdevice, nv_gpu.GT200_DEBUGGER, debugger_params)
    elif not video: self.iface.rm_alloc(gpfifo, self.iface.dma_class)
    else: self.iface.rm_alloc(gpfifo, self.iface.viddec_class)

    if channel_group == self.nvdevice:
      self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_BIND, nv_gpu.NVA06F_CTRL_BIND_PARAMS(engineType=params.engineType))
      self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))

    ws_token_params = self.iface.rm_control(gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
      nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1))
    if ctxshare != 0: self.iface.setup_gpfifo_vm(gpfifo)

    return GPFifo(ring=gpfifo_area.cpu_view().view(offset, entries*8, fmt='Q'), entries_count=entries, token=ws_token_params.workSubmitToken,
                  gpput=gpfifo_area.cpu_view().view(offset + entries*8 + getattr(nv_gpu.AmpereAControlGPFifo, 'GPPut').offset, fmt='I'))

  def _query_gpu_info(self, *reqs):
    nvrs = [getattr(nv_gpu,'NV2080_CTRL_GR_INFO_INDEX_'+r.upper(), getattr(nv_gpu,'NV2080_CTRL_GR_INFO_INDEX_LITTER_'+r.upper(), None)) for r in reqs]

    if self.is_nvd():
      x = self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO,
        nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS())
      return [x.engineInfo[0].infoList[nvr].data for nvr in nvrs]

    infos = (nv_gpu.NV2080_CTRL_GR_INFO*len(nvrs))(*[nv_gpu.NV2080_CTRL_GR_INFO(index=nvr) for nvr in nvrs])
    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_GR_GET_INFO,
      nv_gpu.NV2080_CTRL_GR_GET_INFO_PARAMS(grInfoListSize=len(infos), grInfoList=ctypes.addressof(infos)))
    return [x.data for x in infos]

  def _setup_gpfifos(self):
    self.slm_per_thread, self.shader_local_mem = 0, None

    # Set windows addresses to not collide with other allocated buffers.
    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000

    NVComputeQueue().setup(compute_class=self.iface.compute_class, local_mem_window=self.local_mem_window, shared_mem_window=self.shared_mem_window) \
                    .signal(self.timeline_signal, self.next_timeline()).submit(self)

    NVCopyQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                 .setup(copy_class=self.iface.dma_class) \
                 .signal(self.timeline_signal, self.next_timeline()).submit(self)

    self.synchronize()

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required: return

    self.slm_per_thread, old_slm_per_thread = round_up(required, 32), self.slm_per_thread
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    self.shader_local_mem, ok = self._realloc(self.shader_local_mem, round_up(bytes_per_tpc*self.num_tpc_per_gpc*self.num_gpcs, 0x20000))

    # Realloc failed, restore the old value.
    if not ok: self.slm_per_thread = old_slm_per_thread

    cast(NVComputeQueue, NVComputeQueue().wait(self.timeline_signal, self.timeline_value - 1)) \
                                         .setup(local_mem=self.shader_local_mem.va_addr, local_mem_tpc_bytes=bytes_per_tpc) \
                                         .signal(self.timeline_signal, self.next_timeline()).submit(self)

  def _ensure_has_vid_hw(self, w, h):
    if self.iface.viddec_class is None: raise RuntimeError(f"{self.device} Video decoder class not available.")

    coloc_size = round_up((round_up(h, 64) * round_up(h, 64)) + (round_up(w, 64) * round_up(h, 64) // 16), 2 << 20)
    self.intra_top_off = round_up(h, 64) * (608 + 4864 + 152 + 2000)
    intra_unk_size = ((2 << 20) if self.iface.viddec_class >= nv_gpu.NVCFB0_VIDEO_DECODER else 0)
    self.intra_unk_off = (round_up(self.intra_top_off, 0x10000) + (64 << 10)) if intra_unk_size > 0 else None
    filter_size = round_up(round_up(self.intra_top_off, 0x10000) + (64 << 10) + intra_unk_size, 2 << 20)

    if not hasattr(self, 'vid_gpfifo'):
      self.vid_gpfifo = self._new_gpu_fifo(self.gpfifo_area, 0, self.nvdevice, offset=0x200000, entries=2048, compute=False, video=True)
      self.vid_coloc_buf, self.vid_filter_buf = self.allocator.alloc(coloc_size), self.allocator.alloc(filter_size)
      self.vid_stat_buf = self.allocator.alloc(0x1000)
      NVVideoQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                    .setup(copy_class=self.iface.viddec_class) \
                    .signal(self.timeline_signal, self.next_timeline()).submit(self)
    else:
      if coloc_size > self.vid_coloc_buf.size: self.vid_coloc_buf, _ = self._realloc(self.vid_coloc_buf, coloc_size, force=True)
      if filter_size > self.vid_filter_buf.size: self.vid_filter_buf, _ = self._realloc(self.vid_filter_buf, filter_size, force=True)

  def invalidate_caches(self):
    if self.is_nvd(): self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_BUS_FLUSH_WITH_SYSMEMBAR, None)
    else:
      self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE, nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS(
        flags=((nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_YES << 2) | (nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_INVALIDATE_YES << 3) |
              (nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE_FULL_CACHE << 4))))

  def on_device_hang(self):
    # Prepare fault report.
    # TODO: Restore the GPU using NV83DE_CTRL_CMD_CLEAR_ALL_SM_ERROR_STATES if needed.

    report = []
    sm_errors = self.iface.rm_control(self.debugger, nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES,
      nv_gpu.NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS(hTargetChannel=self.debug_channel, numSMsToRead=100))

    if sm_errors.mmuFault.valid:
      mmu = self.iface.rm_control(self.debugger, nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO,
        nv_gpu.NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS())
      for i in range(mmu.count):
        pfinfo = mmu.mmuFaultInfoList[i]
        report += [f"MMU fault: 0x{pfinfo.faultAddress:X} | {NV_PFAULT_FAULT_TYPE[pfinfo.faultType]} | {NV_PFAULT_ACCESS_TYPE[pfinfo.accessType]}"]
    else:
      for i, e in enumerate(sm_errors.smErrorStateArray):
        if e.hwwGlobalEsr or e.hwwWarpEsr: report += [f"SM {i} fault: esr={e.hwwGlobalEsr} warp_esr={e.hwwWarpEsr:#x} warp_pc={e.hwwWarpEsrPc64:#x}"]

    raise RuntimeError("\n".join(report))

  def _prof_init(self):
    assert not self.is_nvd()

    self.profiler = self.iface.rm_alloc(self.subdevice, nv_gpu.MAXWELL_PROFILER_DEVICE,
      nv_gpu.NVB2CC_ALLOC_PARAMETERS(hClientTarget=self.iface.root, hContextTarget=self.channel_group))

    power_params = nv_gpu.struct_NVB0CC_CTRL_POWER_REQUEST_FEATURES_PARAMS(controlMask=(nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_ELCG_DISABLE << 0) | \
      (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_BLCG_DISABLE << 2) | (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_ELPG_DISABLE << 6) | \
      (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_IDLE_SLOWDOWN_DISABLE << 8) | (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_VAT_DISABLE << 10))
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES, power_params)

    self.pma_buf = self.iface.alloc(getenv("PMA_BUFFER_SIZE", 512) << 20, uncached=True, cpu_cached=True, cpu_access=True)
    self.pma_bytes = self.iface.alloc(0x1000, uncached=True, cpu_cached=True, read_only=True)
    self.pma_rptr = 0

    pma_stream = nv_gpu.struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS(hMemPmaBuffer=self.pma_buf.meta.hMemory,
      pmaBufferSize=self.pma_buf.size, hMemPmaBytesAvailable=self.pma_bytes.meta.hMemory, pmaBufferVA=self.pma_buf.va_addr)
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM, pma_stream)

    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY, nv_gpu.struct_NVB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS(ctxsw=0))
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_RESERVE_PM_AREA_PC_SAMPLER)
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_BIND_PM_RESOURCES)

    self._prof_setup_pc_sampling()

  def _prof_setup_pc_sampling(self):
    is_bw = self.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A
    PMASYS_BASE, PMAGPC_BASE, GR_GPC_BASE, GPC_BASE = (0x2b1000, 0x2b0000, 0x424000, 0x200000) if is_bw else (0x24a000, 0x244000, 0x419800, 0x180000)

    tpc_masks = [m for i in range(self.num_gpcs) if (m:=self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_GR_GET_TPC_MASK,
      nv_gpu.NV2080_CTRL_GR_GET_TPC_MASK_PARAMS(gpcId=i)).tpcMask) > 0]
    tpc_cnt = [bin(mask).count('1') for mask in tpc_masks]

    # enables pma on gpc
    if not is_bw: self.reg_ops(*[(PMAGPC_BASE + gpc * 0x200, 0x100, 0x100) for gpc in range(len(tpc_masks))])

    # sets streaming bw for each gpc
    hs = nv_gpu.struct_NVB0CC_CTRL_HS_CREDITS_PARAMS(pmaChannelIdx=0, numEntries=len(tpc_masks))
    for i, mask in enumerate(tpc_masks):
      hs.creditInfo[i] = nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO(
        chipletType=nv_gpu.NVB0CC_CHIPLET_TYPE_GPC, chipletIndex=i, numCredits=bin(mask).count('1'))
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_SET_HS_CREDITS, hs)

    if is_bw:
      # enables pma on gpcs
      self.reg_ops(*[op for i in range(3) for op in [(PMASYS_BASE + 0x128 + i*8, 480), (PMASYS_BASE + 0x12c + i*8, 0x80000000)]])
      self.reg_ops((PMAGPC_BASE + 0xa24, 0x04000001), (PMAGPC_BASE + 0xa10, 0x80000002))
      self.reg_ops(*[(GPC_BASE + gpc * 0x4000 + 0x200 + tpc * 0x200 + reg, 0)
                     for gpc in range(len(tpc_masks)) for tpc in range(tpc_cnt[gpc]) for reg in [0x100, 0x108, 0x110, 0x120]])

      def SM_REG(gpc, tpc, sm, reg): return GPC_BASE + gpc * 0x4000 + 0x800 + (tpc * self.num_sm_per_tpc + sm) * 0x200 + reg
    else:
      self.reg_ops(*[(PMASYS_BASE + 0x65c + off * 4, 0xffffffff) for off in range(self.num_gpcs * 2)])
      self.reg_ops((PMASYS_BASE + 0x620, 0x2000007))

      def SM_REG(gpc, tpc, sm, reg): return GPC_BASE + gpc * 0x4000 + (self.num_tpc_per_gpc - tpc_cnt[gpc] + tpc) * 0x200 + [0x400, 0x1000][sm] + reg

    # enable pc sampling for the context
    self.reg_ops((GR_GPC_BASE + 0x304, 0x80808a))

    # sm config and enable
    self.reg_ops(*[op for gpc in range(len(tpc_masks)) for tpc in range(tpc_cnt[gpc]) for sm in range(self.num_sm_per_tpc) for op in [
      (SM_REG(gpc, tpc, sm, 0x128), (gpc << 5) | (tpc << 1) | sm), # enumeration. NOTE: different from cuda
      (SM_REG(gpc, tpc, sm, 0x40), 0x19181716), (SM_REG(gpc, tpc, sm, 0x48), 0x1d1c1b1a), (SM_REG(gpc, tpc, sm, 0x50), 0x1e201f), # unk, counters?
      (SM_REG(gpc, tpc, sm, 0xec), 0x1), (SM_REG(gpc, tpc, sm, 0x6c), 0x2), (SM_REG(gpc, tpc, sm, 0x9c), 0x5),
      (SM_REG(gpc, tpc, sm, 0x108), 0xa0 if is_bw else 0x20), *([(SM_REG(gpc, tpc, sm, 0x120), 0x100000)] if is_bw else [])]])
    self.reg_ops((GR_GPC_BASE + 0x3dc, 0x1), reg_type=1)

  def reg_ops(self, *ops, reg_type=0, op=nv_gpu.NV2080_CTRL_GPU_REG_OP_WRITE_32):
    for i in range(0, len(ops), 124):
      params = nv_gpu.struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS(regOpCount=len(chunk:=ops[i:i+124]))
      for j, (off, val, *rest) in enumerate(chunk):
        params.regOps[j] = nv_gpu.struct_NV2080_CTRL_GPU_REG_OP(regOp=op, regType=reg_type,
          regOffset=off, regValueLo=val, regAndNMaskLo=rest[0] if rest else 0xffffffff)
      with contextlib.suppress(RuntimeError): self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_EXEC_REG_OPS, params)

  def _prof_readback(self) -> bytes|None:
    params = self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT,
      nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS(bUpdateAvailableBytes=1, bWait=1))

    if params.bOverflowStatus: raise RuntimeError("PMA profiler: buffer overflow detected")
    if params.bytesAvailable == 0: return None

    start, end = self.pma_rptr, self.pma_rptr + params.bytesAvailable
    pma_data = self.pma_buf.cpu_view()[start:min(end, self.pma_buf.size)] + self.pma_buf.cpu_view()[:max(0, end - self.pma_buf.size)]
    self.pma_rptr = end % self.pma_buf.size

    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT,
      nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS(bytesConsumed=params.bytesAvailable))
    return pma_data

  def device_props(self): return {'arch': self.arch, 'sm_version': self.sm_version}
