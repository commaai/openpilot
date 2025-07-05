from __future__ import annotations
import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, weakref, traceback
assert sys.platform != 'win32'
from typing import cast, Union, ClassVar
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQProgram, HCQSignal, BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface, FileIOInterface, MOCKGPU
from tinygrad.uop.ops import sint
from tinygrad.device import BufferSpec
from tinygrad.helpers import getenv, mv_address, round_up, data64, data64_le, prod, OSX, to_mv, hi32, lo32
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import NVRenderer
from tinygrad.runtime.support.compiler_cuda import CUDACompiler, PTXCompiler, PTX, NVPTXCompiler, NVCompiler
from tinygrad.runtime.autogen import nv_gpu, pci
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.system import System, PCIIfaceBase
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401 # pylint: disable=unused-import

def get_error_str(status): return f"{status}: {nv_gpu.nv_status_codes.get(status, 'Unknown error')}"

NV_PFAULT_FAULT_TYPE = {dt:name for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_FAULT_TYPE_")}
NV_PFAULT_ACCESS_TYPE = {dt:name.split("_")[-1] for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_ACCESS_TYPE_")}

def nv_iowr(fd:FileIOInterface, nr, args):
  ret = fd.ioctl((3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

def uvm_ioctl(cmd, sttyp, fd:FileIOInterface, **kwargs):
  ret = fd.ioctl(cmd, made:=sttyp(**kwargs))
  if ret != 0: raise RuntimeError(f"ioctl(uvm) returned {ret}")
  if made.rmStatus != 0: raise RuntimeError(f"uvm_ioctl returned {get_error_str(made.rmStatus)}")
  return made

def make_uvm_type():
  return type("NVUVM", (object,), {name.replace("UVM_", "").lower(): functools.partial(uvm_ioctl, dt, getattr(nv_gpu, name+"_PARAMS"))
                                   for name,dt in nv_gpu.__dict__.items() if name.startswith("UVM_") and nv_gpu.__dict__.get(name+"_PARAMS")})
uvm = make_uvm_type()

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

class NVSignal(HCQSignal):
  def __init__(self, base_buf:HCQBuffer|None=None, **kwargs):
    super().__init__(base_buf, **kwargs, timestamp_divider=1000, dev_t=NVDevice)

class NVCommandQueue(HWQueue[NVSignal, 'NVDevice', 'NVProgram', 'NVArgsState']):
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

  def wait(self, signal:NVSignal, value:sint=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value), (3 << 0) | (1 << 24)) # ACQUIRE | PAYLOAD_SIZE_64BIT
    self.active_qmd = None
    return self

  def timestamp(self, signal:NVSignal): return self.signal(signal, 0)

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
      cmdq_addr = dev.cmdq_allocator.alloc(len(self._q) * 4)
      cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
      dev.cmdq[cmdq_wptr : cmdq_wptr + len(self._q)] = array.array('I', self._q)

    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr//4 << 2) | (len(self._q) << 42) | (1 << 41)
    gpfifo.controls.GPPut = (gpfifo.put_value + 1) % gpfifo.entries_count

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

    qmd = QMD(dev=prg.dev, addr=cast(int, qmd_buf.va_addr)) # Save qmd for later update

    self.bind_sints_to_mem(*global_size, mem=qmd_buf.cpu_view(), fmt='I', offset=qmd.field_offset('cta_raster_width' if qmd.ver<4 else 'grid_width'))
    self.bind_sints_to_mem(*(local_size[:2]), mem=qmd_buf.cpu_view(), fmt='H', offset=qmd.field_offset('cta_thread_dimension0'))
    self.bind_sints_to_mem(local_size[2], mem=qmd_buf.cpu_view(), fmt='B', offset=qmd.field_offset('cta_thread_dimension2'))
    qmd.set_constant_buf_addr(0, args_state.buf.va_addr)

    if self.active_qmd is None:
      self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, qmd_buf.va_addr >> 8)
      self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, 9)
    else:
      self.active_qmd.write(dependent_qmd0_pointer=qmd_buf.va_addr >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)

    self.active_qmd, self.active_qmd_buf = qmd, qmd_buf
    return self

  def signal(self, signal:NVSignal, value:sint=0):
    if self.active_qmd is not None:
      for i in range(2):
        if self.active_qmd.read(f'release{i}_enable') == 0:
          self.active_qmd.write(**{f'release{i}_enable': 1})
          self.bind_sints_to_mem(signal.value_addr, mem=self.active_qmd_buf.cpu_view(), fmt='Q', mask=0xfffffffff,
            offset=self.active_qmd.field_offset(f'release{i}_address_lower' if self.active_qmd.ver<4 else f'release_semaphore{i}_addr_lower'))
          self.bind_sints_to_mem(value, mem=self.active_qmd_buf.cpu_view(), fmt='Q',
            offset=self.active_qmd.field_offset(f'release{i}_payload_lower' if self.active_qmd.ver<4 else f'release_semaphore{i}_payload_lower'))
          return self

    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             (1 << 0) | (1 << 20) | (1 << 24) | (1 << 25)) # RELEASE | RELEASE_WFI | PAYLOAD_SIZE_64BIT | RELEASE_TIMESTAMP
    self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)
    self.active_qmd = None
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.compute_gpfifo)

class NVCopyQueue(NVCommandQueue):
  def copy(self, dest:sint, src:sint, copy_size:int):
    for off in range(0, copy_size, step:=(1 << 31)):
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *data64(src+off), *data64(dest+off))
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, min(copy_size-off, step))
      self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x182) # TRANSFER_TYPE_NON_PIPELINED | DST_MEMORY_LAYOUT_PITCH | SRC_MEMORY_LAYOUT_PITCH
    return self

  def signal(self, signal:NVSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, 0x14)
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.dma_gpfifo)

class NVArgsState(CLikeArgsState):
  def __init__(self, buf:HCQBuffer, prg:NVProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    if MOCKGPU: prg.constbuffer_0[80:82] = [len(bufs), len(vals)]
    super().__init__(buf, prg, bufs, vals=vals, prefix=prg.constbuffer_0)

class NVProgram(HCQProgram):
  def __init__(self, dev:NVDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib

    # For MOCKGPU, the lib is PTX code, so some values are emulated.
    cbuf0_size = 0 if not MOCKGPU else 0x160

    if MOCKGPU: image, sections, relocs = memoryview(bytearray(lib) + b'\x00' * (4 - len(lib)%4)).cast("I"), [], [] # type: ignore
    else: image, sections, relocs = elf_loader(self.lib, force_section_align=128)

    # NOTE: Ensure at least 4KB of space after the program to mitigate prefetch memory faults.
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000) + 0x1000, buf_spec:=BufferSpec(cpu_access=True))

    self.prog_addr, self.prog_sz, self.regs_usage, self.shmem_usage, self.lcmem_usage = self.lib_gpu.va_addr, image.nbytes, 0, 0x400, 0
    self.constbufs: dict[int, tuple[int, int]] = {0: (0, 0x160)} # dict[constbuf index, tuple[va_addr, size]]
    for sh in sections:
      if sh.name == f".nv.shared.{self.name}": self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
      if sh.name == f".text.{self.name}": self.prog_addr, self.prog_sz = self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size
      elif m:=re.match(r'\.nv\.constant(\d+)', sh.name): self.constbufs[int(m.group(1))] = (self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size)
      elif sh.name.startswith(".nv.info"):
        for typ, param, data in self._parse_elf_info(sh):
          if sh.name == f".nv.info.{name}" and param == 0xa: cbuf0_size = struct.unpack_from("IH", data)[1] # EIATTR_PARAM_CBANK
          elif sh.name == ".nv.info" and param == 0x12: self.lcmem_usage = struct.unpack_from("II", data)[1] + 0x240 # EIATTR_MIN_STACK_SIZE
          elif sh.name == ".nv.info" and param == 0x2f: self.regs_usage = struct.unpack_from("II", data)[1] # EIATTR_REGCOUNT

    # Ensure device has enough local memory to run the program
    self.dev._ensure_has_local_memory(self.lcmem_usage)

    # Apply relocs
    for apply_image_offset, rel_sym_offset, typ, _ in relocs:
      # These types are CUDA-specific, applying them here
      if typ == 2: image[apply_image_offset:apply_image_offset+8] = struct.pack('<Q', self.lib_gpu.va_addr + rel_sym_offset) # R_CUDA_64
      elif typ == 0x38: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) & 0xffffffff)
      elif typ == 0x39: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
      else: raise RuntimeError(f"unknown NV reloc {typ}")

    ctypes.memmove(self.lib_gpu.va_addr, mv_address(image), image.nbytes)

    self.constbuffer_0 = [0] * (cbuf0_size // 4)

    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      self.constbuffer_0[188:192], self.constbuffer_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xfffdc0
      qmd = {'qmd_major_version':5, 'qmd_type':nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, 'register_count':self.regs_usage,
        'program_address_upper_shifted4':hi32(self.prog_addr>>4), 'program_address_lower_shifted4':lo32(self.prog_addr>>4),
        'shared_memory_size_shifted7':self.shmem_usage>>7, 'shader_local_memory_high_size_shifted4':self.dev.slm_per_thread>>4}
    else:
      self.constbuffer_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xfffdc0)]
      qmd = {'qmd_major_version':3, 'sm_global_caching_enable':1, 'shader_local_memory_high_size':self.dev.slm_per_thread,
        'program_address_upper':hi32(self.prog_addr), 'program_address_lower':lo32(self.prog_addr), 'shared_memory_size':self.shmem_usage,
        'register_count_v':self.regs_usage}

    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1

    self.qmd:QMD = QMD(dev, **qmd, qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalidate_0=1,
      min_sm_config_shared_mem_size=smem_cfg, target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a,
      program_prefetch_size=min(self.prog_sz>>8, 0x1ff), sass_version=dev.sass_version,
      program_prefetch_addr_upper_shifted=self.prog_addr>>40, program_prefetch_addr_lower_shifted=self.prog_addr>>8)

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

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    if prod(local_size) > 1024 or self.max_threads < prod(local_size) or self.lcmem_usage > cast(NVDevice, self.dev).slm_per_thread:
      raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {self.max_threads=}")
    if any(cur > mx for cur,mx in zip(global_size, [2147483647, 65535, 65535])) or any(cur > mx for cur,mx in zip(local_size, [1024, 1024, 64])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

class NVAllocator(HCQAllocator['NVDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev.iface.alloc(size, cpu_access=options.cpu_access, host=options.host)

  def _free(self, opaque:HCQBuffer, options:BufferSpec):
    try:
      self.dev.synchronize()
      self.dev.iface.free(opaque)
    except AttributeError: pass

  def map(self, buf:HCQBuffer): self.dev.iface.map(buf._base if buf._base is not None else buf)

@dataclass
class GPFifo:
  ring: MMIOInterface
  controls: nv_gpu.AmpereAControlGPFifo
  entries_count: int
  token: int
  put_value: int = 0

MAP_FIXED, MAP_NORESERVE = 0x10, 0x400
class NVKIface:
  root = None
  fd_ctl: FileIOInterface
  fd_uvm: FileIOInterface
  gpus_info: Union[list, ctypes.Array] = []

  # TODO: Need a proper allocator for va addresses
  # 0x1000000000 - 0x2000000000, reserved for system/cpu mappings
  # VA space is 48bits.
  low_uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=0x1000000000, base=0x8000000000 if OSX else 0x1000000000, wrap=False)
  uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=(1 << 48) - 1, base=low_uvm_vaddr_allocator.base + low_uvm_vaddr_allocator.size, wrap=False)
  host_object_enumerator: int = 0x1000

  def __init__(self, dev, device_id):
    if NVKIface.root is None:
      NVKIface.fd_ctl = FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.fd_uvm = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      self.fd_uvm_2 = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.root = self.rm_alloc(0, nv_gpu.NV01_ROOT_CLIENT, None, root=0)
      uvm.initialize(self.fd_uvm)
      with contextlib.suppress(RuntimeError): uvm.mm_initialize(self.fd_uvm_2, uvmFd=self.fd_uvm.fd) # this error is okay, CUDA hits it too

      nv_iowr(NVKIface.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, gpus_info:=(nv_gpu.nv_ioctl_card_info_t*64)())
      visible_devices = [int(x) for x in (getenv('VISIBLE_DEVICES', getenv('CUDA_VISIBLE_DEVICES', ''))).split(',') if x.strip()]
      NVKIface.gpus_info = [gpus_info[x] for x in visible_devices] if visible_devices else gpus_info

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
      paramsSize=ctypes.sizeof(params), params=ctypes.cast(ctypes.byref(params), ctypes.c_void_p) if params is not None else None))
    if made.status != 0: raise RuntimeError(f"rm_control returned {get_error_str(made.status)}")
    return params

  def setup_usermode(self):
    clsinfo = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS(numClasses=100,
      classList=mv_address(classlist:=memoryview(bytearray(100 * 4)).cast('I'))))
    self.nvclasses = {classlist[i] for i in range(clsinfo.numClasses)}
    self.usermode_class:int = next(c for c in [nv_gpu.HOPPER_USERMODE_A, nv_gpu.TURING_USERMODE_A] if c in self.nvclasses)
    self.gpfifo_class:int = next(c for c in [nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_CHANNEL_GPFIFO_A] if c in self.nvclasses)
    self.compute_class:int = next(c for c in [nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.ADA_COMPUTE_A, nv_gpu.AMPERE_COMPUTE_B] if c in self.nvclasses)
    self.dma_class:int = next(c for c in [nv_gpu.BLACKWELL_DMA_COPY_B, nv_gpu.AMPERE_DMA_COPY_B] if c in self.nvclasses)

    usermode = self.rm_alloc(self.dev.subdevice, self.usermode_class)
    return usermode, MMIOInterface(self._gpu_map_to_cpu(usermode, mmio_sz:=0x10000, flags=2), mmio_sz, fmt='I')

  def setup_vm(self, vaspace):
    self.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO, raw_uuid:=nv_gpu.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(
      flags=nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16))
    self.gpu_uuid = nv_gpu.struct_nv_uuid(uuid=(ctypes.c_ubyte*16)(*[raw_uuid.data[i] for i in range(16)]))

    uvm.register_gpu(self.fd_uvm, rmCtrlFd=-1, gpu_uuid=self.gpu_uuid)
    uvm.register_gpu_vaspace(self.fd_uvm, gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hVaSpace=vaspace)

    for dev in cast(list[NVDevice], self.dev.devices):
      try: uvm.enable_peer_access(self.fd_uvm, gpuUuidA=self.gpu_uuid, gpuUuidB=dev.iface.gpu_uuid)
      except RuntimeError as e: raise RuntimeError(f"{e}. Make sure GPUs #{self.gpu_minor} & #{dev.iface.gpu_minor} have P2P enabled.") from e

  def setup_gpfifo_vm(self, gpfifo):
    uvm.register_channel(self.fd_uvm, gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root,
                         hChannel=gpfifo, base=self._alloc_gpu_vaddr(0x4000000, force_low=True), length=0x4000000)

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

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, map_flags=0) -> HCQBuffer:
    # Uncached memory is "system". Use huge pages only for gpu memory.
    page_size = (4 << (12 if OSX else 10)) if uncached or host else ((2 << 20) if size >= (8 << 20) else (4 << (12 if OSX else 10)))
    size = round_up(size, page_size)
    va_addr = self._alloc_gpu_vaddr(size, alignment=page_size, force_low=cpu_access)

    if host:
      va_addr = FileIOInterface.anon_mmap(va_addr, size, mmap.PROT_READ | mmap.PROT_WRITE, MAP_FIXED | mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, 0)

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
            | ((nv_gpu.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB if page_size > 0x1000 else 0) << 20) | nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC

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

    uvm.free(self.fd_uvm, base=cast(int, mem.va_addr), length=mem.size)
    if mem.meta.has_cpu_mapping: FileIOInterface.munmap(cast(int, mem.va_addr), mem.size)

  def _gpu_uvm_map(self, va_base, size, mem_handle, create_range=True, has_cpu_mapping=False) -> HCQBuffer:
    if create_range: uvm.create_external_range(self.fd_uvm, base=va_base, length=size)
    attrs = (nv_gpu.struct_c__SA_UvmGpuMappingAttributes*256)(nv_gpu.struct_c__SA_UvmGpuMappingAttributes(gpuUuid=self.gpu_uuid, gpuMappingType=1))

    # NOTE: va_addr is set to make rawbufs compatible with HCQBuffer protocol.
    return HCQBuffer(va_base, size, meta=uvm.map_external_allocation(self.fd_uvm, base=va_base, length=size, rmCtrlFd=self.fd_ctl.fd,
      hClient=self.root, hMemory=mem_handle, gpuAttributesCount=1, perGpuAttributes=attrs,
      mapped_gpu_ids=[self.gpu_uuid], has_cpu_mapping=has_cpu_mapping),
      view=MMIOInterface(va_base, size, fmt='B') if has_cpu_mapping else None)

  def map(self, mem:HCQBuffer):
    if self.gpu_uuid in mem.meta.mapped_gpu_ids: return
    mem.meta.mapped_gpu_ids.append(self.gpu_uuid)
    self._gpu_uvm_map(mem.va_addr, mem.size, mem.meta.hMemory, create_range=False)

  def _alloc_gpu_vaddr(self, size, alignment=(4 << 10), force_low=False):
    return NVKIface.low_uvm_vaddr_allocator.alloc(size, alignment) if force_low else NVKIface.uvm_vaddr_allocator.alloc(size, alignment)

class PCIIface(PCIIfaceBase):
  def __init__(self, dev, dev_id):
    super().__init__(dev, dev_id, vendor=0x10de, devices=[0x2684], bars=[0, 1], vram_bar=1,
      va_start=NVMemoryManager.va_allocator.base, va_size=NVMemoryManager.va_allocator.size)
    System.reserve_hugepages(64)

    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.dev_impl:NVDev = NVDev(self.pci_dev.pcibus, self.pci_dev.map_bar(0, fmt='I'), self.pci_dev.map_bar(1),
      self.pci_dev.read_config(pci.PCI_VENDOR_ID, 4), self.pci_dev.read_config(pci.PCI_SUBSYSTEM_VENDOR_ID, 4),
      self.pci_dev.read_config(pci.PCI_REVISION_ID, 1), self.pci_dev.bar_info)
    self.root, self.gpu_instance, self.p2p_base_addr = 0xc1000000, 0, self.pci_dev.bar_info[1][0]
    self.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())

    # Setup classes for the GPU
    self.gpfifo_class, self.compute_class, self.dma_class = nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.ADA_COMPUTE_A, nv_gpu.AMPERE_DMA_COPY_B

  def setup_usermode(self): return 0xce000000, self.pci_dev.map_bar(bar=0, fmt='I', off=0xbb0000, size=0x10000)
  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass

  def rm_alloc(self, parent, clss, params=None, root=None) -> int: return self.dev_impl.gsp.rpc_rm_alloc(parent, clss, params, self.root)
  def rm_control(self, obj, cmd, params=None):
    res = self.dev_impl.gsp.rpc_rm_control(obj, cmd, params, self.root)
    return type(params).from_buffer_copy(res) if params is not None else None

class NVDevice(HCQCompiled[NVSignal]):
  devices: ClassVar[list[HCQCompiled]] = []
  signal_pages: ClassVar[list[HCQBuffer]] = []
  signal_pool: ClassVar[list[HCQBuffer]] = []

  def is_nvd(self) -> bool: return isinstance(self.iface, PCIIface)

  def _select_iface(self):
    if len(nm:=getenv("NV_IFACE", "")) > 0: return getattr(sys.modules[__name__], f"{nm.upper()}Iface")(self, self.device_id)

    errs:str = ""
    for iface_t in (NVKIface, PCIIface):
      try: return iface_t(self, self.device_id)
      except Exception: errs += f"\n{iface_t.__name__}: {traceback.format_exc()}"
    raise RuntimeError(f"Cannot find a usable interface for NV:{self.device_id}:\n{errs}")

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = self._select_iface()

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root,
                                                   vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES)
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())
    self.usermode, self.gpu_mmio = self.iface.setup_usermode()

    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff,
      flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) | \
             (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX))))

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A, vaspace_params)

    self.iface.setup_vm(vaspace)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    channel_group = self.iface.rm_alloc(self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, channel_params)

    gpfifo_area = self.iface.alloc(0x200000, contiguous=True, cpu_access=True, map_flags=0x10d0000)

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = self.iface.rm_alloc(channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A, ctxshare_params)

    self.compute_gpfifo = self._new_gpu_fifo(gpfifo_area, ctxshare, channel_group, offset=0, entries=0x10000, compute=True)
    self.dma_gpfifo = self._new_gpu_fifo(gpfifo_area, ctxshare, channel_group, offset=0x100000, entries=0x10000, compute=False)
    self.iface.rm_control(channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))

    self.cmdq_page:HCQBuffer = self.iface.alloc(0x200000, cpu_access=True)
    self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=cast(int, self.cmdq_page.va_addr), wrap=True)
    self.cmdq = MMIOInterface(cast(int, self.cmdq_page.va_addr), 0x200000, fmt='I')

    self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version = self._query_gpu_info('num_gpcs',
      'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version')

    # FIXME: no idea how to convert this for blackwells
    self.arch: str = "sm_120" if self.sm_version==0xa04 else f"sm_{(self.sm_version>>8)&0xff}{(val>>4) if (val:=self.sm_version&0xff) > 0xf else val}"
    self.sass_version = ((self.sm_version & 0xf00) >> 4) | (self.sm_version & 0xf)

    compiler_t = (PTXCompiler if PTX else CUDACompiler) if MOCKGPU else (NVPTXCompiler if PTX else NVCompiler)
    super().__init__(device, NVAllocator(self), PTXRenderer(self.arch, device="NV") if PTX else NVRenderer(self.arch), compiler_t(self.arch),
                     functools.partial(NVProgram, self), NVSignal, NVComputeQueue, NVCopyQueue)

    self._setup_gpfifos()

  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, compute=False) -> GPFifo:
    notifier = self.iface.alloc(48 << 20, uncached=True)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(hObjectError=notifier.meta.hMemory, hObjectBuffer=gpfifo_area.meta.hMemory,
      gpFifoOffset=gpfifo_area.va_addr+offset, gpFifoEntries=entries, hContextShare=ctxshare,
      hUserdMemory=(ctypes.c_uint32*8)(gpfifo_area.meta.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset))
    gpfifo = self.iface.rm_alloc(channel_group, self.iface.gpfifo_class, params)

    if compute:
      self.debug_compute_obj, self.debug_channel = self.iface.rm_alloc(gpfifo, self.iface.compute_class), gpfifo
      debugger_params = nv_gpu.NV83DE_ALLOC_PARAMETERS(hAppClient=self.iface.root, hClass3dObject=self.debug_compute_obj)
      self.debugger = self.iface.rm_alloc(self.nvdevice, nv_gpu.GT200_DEBUGGER, debugger_params)
    else: self.iface.rm_alloc(gpfifo, self.iface.dma_class)

    ws_token_params = self.iface.rm_control(gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
      nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1))
    self.iface.setup_gpfifo_vm(gpfifo)

    return GPFifo(ring=MMIOInterface(gpfifo_area.va_addr + offset, entries*8, fmt='Q'), entries_count=entries, token=ws_token_params.workSubmitToken,
                  controls=nv_gpu.AmpereAControlGPFifo.from_address(gpfifo_area.va_addr + offset + entries * 8))

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
                    .signal(self.timeline_signal, self.timeline_value).submit(self)

    cast(NVCopyQueue, NVCopyQueue().wait(self.timeline_signal, self.timeline_value)) \
                                   .setup(copy_class=self.iface.dma_class) \
                                   .signal(self.timeline_signal, self.timeline_value + 1).submit(self)

    self.timeline_value += 2

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required or ((maxlm:=getenv("NV_MAX_LOCAL_MEMORY_PER_THREAD")) > 0 and required >= maxlm): return

    self.slm_per_thread, old_slm_per_thread = round_up(required, 32), self.slm_per_thread
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    self.shader_local_mem, ok = self._realloc(self.shader_local_mem, round_up(bytes_per_tpc*self.num_tpc_per_gpc*self.num_gpcs, 0x20000))

    # Realloc failed, restore the old value.
    if not ok: self.slm_per_thread = old_slm_per_thread

    cast(NVComputeQueue, NVComputeQueue().wait(self.timeline_signal, self.timeline_value - 1)) \
                                         .setup(local_mem=self.shader_local_mem.va_addr, local_mem_tpc_bytes=bytes_per_tpc) \
                                         .signal(self.timeline_signal, self.next_timeline()).submit(self)

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
