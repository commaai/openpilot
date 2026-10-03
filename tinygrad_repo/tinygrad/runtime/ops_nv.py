from __future__ import annotations
import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, itertools
assert sys.platform != 'win32'
from typing import Any
from dataclasses import dataclass, replace
from tinygrad.runtime.support.hcq2 import HWQueue, encode_submit, patch, to_name, unwrap_view, make_submit, timeline, HCQInfo, lower_call, hcq_link
from tinygrad.runtime.support.hcq2 import layout_args
from tinygrad.runtime.support.hcq import MMIOInterface, FileIOInterface, BumpAllocator, hcq_filter_visible_devices
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops, lower_and_compile, run_linear
from tinygrad.device import BufferStorage, Buffer, BufferSpec, Allocator, Compiled, Device, TinyELF
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import getenv, mv_address, round_up, data64, data64_le, prod, OSX, PROFILE, ContextVar, VIZ
from tinygrad.helpers import ProfileEvent, unwrap
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import CUDARenderer, NVCCRenderer
from tinygrad.runtime.autogen import nv_570, nv_580, nv_610, mesa
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.system import PCIIfaceBase, MAP_FIXED
from tinygrad.renderer.nir import NAKRenderer
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401 # pylint: disable=unused-import

nv_gpu = nv_570 # default to 570

PMA = ContextVar("PMA", abs(VIZ.value)>=2)

@dataclass(frozen=True)
class ProfilePMAEvent(ProfileEvent): device:str; kern:str; blob:bytes; exec_tag:int; profile_key:bytes|None=None # noqa: E702

def hilo(addr:UOp) -> tuple[UOp, UOp]: return (addr >> 32).cast(dtypes.uint32), addr.cast(dtypes.uint32)

def get_error_str(status): return f"{status}: {nv_gpu.nv_status_codes.get(status, 'Unknown error')}"

NV_PFAULT_FAULT_TYPE = {dt:name for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_FAULT_TYPE_")}
NV_PFAULT_ACCESS_TYPE = {dt:name.split("_")[-1] for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_ACCESS_TYPE_")}

def nv_flags(reg, **kwargs): return functools.reduce(int.__or__, ((getattr(nv_gpu, f"{reg}_{k}_{v}".upper()) if isinstance(v, str) else v) <<
  getattr(nv_gpu, f"{reg}_{k}".upper())[1] for k, v in kwargs.items()), 0)

def nv_iowr(fd:FileIOInterface, nr, args, cmd=None):
  ret = fd.ioctl(cmd or ((3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF)), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

def nvm(subc:int, mthd:int, *vals, typ=2) -> list:
  return [(typ << 28) | (sum(v.dtype.itemsize // 4 if isinstance(v, UOp) else 1 for v in vals) << 16) | (subc << 13) | (mthd >> 2), *vals]

class QMD:
  fields: dict[str, dict[str, tuple[int, int]]] = {}

  def __init__(self, dev:NVDevice, blob:bytearray|None=None):
    self.ver, self.sz = (5, 0x60) if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)

    # Init fields from module
    if (pref:="NVCEC0_QMDV05_00" if self.ver == 5 else "NVC6C0_QMDV03_00") not in QMD.fields:
      QMD.fields[pref] = {**{name[len(pref)+1:]: dt for name,dt in nv_gpu.__dict__.items() if name.startswith(pref) and isinstance(dt, tuple)},
        **{name[len(pref)+1:]+f"_{i}": dt(i) for name,dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith(pref) and callable(dt)}}

    self.mv, self.pref = (bytearray(self.sz * 4) if blob is None else blob), pref
    self.patches:dict[int, UOp] = {}

  def read(self, k:str) -> int:
    hi, lo = QMD.fields[self.pref][k.upper()]
    return (int.from_bytes(self.mv[lo//8:hi//8+1], "little") >> (lo % 8)) & ((1 << (hi - lo + 1)) - 1)

  def write(self, **kwargs:int|UOp):
    for k, v in kwargs.items():
      hi, lo = QMD.fields[self.pref][k.upper()]
      if isinstance(v, UOp):
        assert lo % 8 == 0, f"{k} is not byte aligned"
        self.patches[lo // 8] = v.ccast(next(t for t in (dtypes.uint64, dtypes.uint32, dtypes.uint16, dtypes.uint8) if t.itemsize * 8 <= hi - lo + 1))
      else:
        if v >> (hi - lo + 1): raise ValueError(f"{k}={v:#x} does not fit")
        mask, num = ((1 << (hi - lo + 1)) - 1) << (lo % 8), int.from_bytes(self.mv[lo//8:hi//8+1], "little")
        self.mv[lo//8:hi//8+1] = ((num & ~mask) | (v << (lo % 8))).to_bytes(hi//8 - lo//8 + 1, "little")

  def set_addr(self, name:str, addr:UOp, sfx:str=""): self.write(**{f"{name}_lower{sfx}": addr, f"{name}_upper{sfx}": addr >> 32})
  def set_constant_buf_addr(self, i:int, addr:UOp):
    self.set_addr("constant_buffer_addr", addr >> (6 if self.ver >= 4 else 0), f"_shifted6_{i}" if self.ver >= 4 else f"_{i}")
  def set_program_addr(self, addr:UOp):
    self.set_addr("program_address", addr >> (4 if self.ver >= 4 else 0), "_shifted4" if self.ver >= 4 else "")
    self.set_addr("program_prefetch_addr", addr >> 8, "_shifted")
  def set_release(self, addr:UOp, payload:UOp, timestamp:bool=False) -> bool:
    if (i:=next((i for i in range(2) if not self.read(f"release{i}_enable")), None)) is None: return False
    self.set_addr(f"release_semaphore{i}_addr" if self.ver >= 4 else f"release{i}_address", addr)
    self.set_addr(f"release_semaphore{i}_payload" if self.ver >= 4 else f"release{i}_payload", payload)
    self.write(**{f"release{i}_enable": 1, f"release_structure_size_{i}" if self.ver >= 4 else f"release{i}_structure_size": 0 if timestamp else 2},
               **({} if self.ver >= 4 else {f"release{i}_payload64b": 1}))
    return True
  @property
  def grid(self) -> tuple[str, ...]:
    return ("grid_width", "grid_height", "grid_depth") if self.ver >= 4 else ("cta_raster_width", "cta_raster_height", "cta_raster_depth")

# *****************
# queues

class NVQueue(HWQueue):
  dev:NVDevice
  q_rewrite = HWQueue.q_rewrite + PatternMatcher([
    (UPat(Ops.INS, arg=("nv", dtypes.void), name="u"), lambda ctx, u: ctx.q(*u.src)),
  ])

  def nvm(self, subc:int, mthd:int, *vals, typ=2): self.q(*nvm(subc, mthd, *vals, typ=typ))

  def sem(self, addr:UOp, value:UOp, **flags:str):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, addr, value.ccast(dtypes.uint64), nv_flags("NVC56F_SEM_EXECUTE", payload_size="64bit", **flags))

  def wait(self, signal:UOp, value:UOp): self.sem(signal.getaddr(self.devs), value, operation="acq_circ_geq")
  def signal(self, signal:UOp, value:UOp): self.release(signal, value)
  def timestamp(self, signal:UOp): self.release(signal, UOp.const(0, dtypes.uint64), timestamp=True)
  def release(self, signal:UOp, value:UOp, timestamp:bool=False):
    self.sem(signal.getaddr(self.devs), value, operation="release", release_wfi="en", release_timestamp="en" if timestamp else "dis")
    if not timestamp: self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)

  def submit(self, cmdbuf:UOp) -> UOp:
    fifo, ib, off = self.dev.fifos[self.queue], *unwrap_view(cmdbuf)

    ring, gpput, doorbell, put, gpentry = [UOp.placeholder((sz,), dt, device=self.devs, volatile=True, tag=to_name(nm, self.queue))
      for nm, dt, sz in (("ring", dtypes.uint64, fifo.entries), ("gpput", dtypes.uint32, 1), ("doorbell", dtypes.uint32, 1),
                         ("put_value", dtypes.uint64, 1), ("gpentry", dtypes.uint64, 1))]
    gpentry = patch(gpentry, [(0, ib.getaddr(self.devs) + UOp.const(off | (cmdbuf.max_numel() // 4 << 42) | (1 << 41), dtypes.uint64))])

    p = put.index(0).load()
    written = UOp.barrier(ring.after(cmdbuf).index((p % fifo.entries).cast(dtypes.int)).store(gpentry.index(0).load()), put.index(0).store(p + 1))
    queued = UOp.barrier(gpput.after(written).index(0).store(((p + 1) % fifo.entries).cast(dtypes.uint32)))
    return doorbell.after(queued).index(0).store(UOp.const(fifo.token, dtypes.uint32))

class NVComputeQueue(NVQueue):
  def __init__(self, ctx, submit):
    super().__init__(ctx, submit)

    progs = [nv_build_program(self.dev, u.body, self.devs)[0] for u in self.lin.src if u.op is Ops.CALL]
    self.qmd_sz = round_up(QMD(self.dev).sz * 4, 256)
    self.stride = self.qmd_sz + max([p.kernargs_size for p in progs], default=0)
    self.qmd_buf = UOp.placeholder((len(progs) * self.stride,), dtypes.uint8, device=self.devs, tag=to_name("qmd", self.queue))
    self.qmds:list[QMD] = []
    self.prev_qmd:QMD|None = None # the launch the next one chains onto

  def wait(self, signal:UOp, value:UOp):
    self.prev_qmd = None
    super().wait(signal, value)

  def release(self, signal:UOp, value:UOp, timestamp:bool=False):
    if self.prev_qmd is None or not self.prev_qmd.set_release(signal.getaddr(self.devs), value, timestamp):
      self.prev_qmd = None
      super().release(signal, value, timestamp)

  def submit(self, cmdbuf:UOp) -> UOp:
    if self.qmds:
      patches = [(i * self.stride + off, w) for i, q in enumerate(self.qmds) for off, w in q.patches.items()]
      cmdbuf = cmdbuf.after(patch(self.qmd_buf, patches, b"".join(q.mv for q in self.qmds)))
    return super().submit(cmdbuf)

  def memory_barrier(self):
    self.prev_qmd = None
    self.nvm(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI,
             nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true"))

  def exec(self, call:UOp, prg:UOp):
    data, lib = nv_build_program(self.dev, prg, self.devs)
    global_size, local_size = prg.arg.global_size, prg.arg.local_size
    if prod(local_size) > 1024 or data.max_threads < prod(local_size):
      raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {data.max_threads=}")
    if any(g > mx for g,mx in zip(global_size, [2147483647, 65535, 65535]) if isinstance(g, int)) or \
       any(l > mx for l,mx in zip(local_size, [1024, 1024, 64])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")

    qmd_addr = self.qmd_buf.getaddr(self.devs) + UOp.const(len(self.qmds) * self.stride, dtypes.uint64)
    qmd = QMD(self.dev, data.qmd.mv.ljust(self.stride, b"\0")) # the program's template, in a slot of its own
    qmd.write(**dict(zip(qmd.grid, global_size)), **{f"cta_thread_dimension{j}": l for j, l in enumerate(local_size)})
    qmd.set_program_addr(lib.getaddr(self.devs) + data.prog_off)
    for j, (off, _) in data.constbufs.items():
      qmd.set_constant_buf_addr(j, qmd_addr + UOp.const(self.qmd_sz, dtypes.uint64) if j == 0 else lib.getaddr(self.devs) + off)
    bufs, vals = [get_call_arg_uops(call)[j] for j in prg.arg.globals], get_call_var_uops(call, prg)
    qmd.mv[self.qmd_sz:(at:=self.qmd_sz + len(data.cbuf_0) * 4)] = array.array('I', data.cbuf_0).tobytes() # constant buffer 0: the driver params
    qmd.patches |= dict(layout_args([b.getaddr(self.devs) for b in bufs] + [v.ccast(dt) for v, dt in zip(vals, data.vars)], at))

    if self.prev_qmd is None:
      if self.dev.pma_enabled: self.nvm(1, nv_gpu.NVC6C0_PM_TRIGGER, 0)
      self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, (qmd_addr >> 8).cast(dtypes.uint32))
      self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE)
    else: self.prev_qmd.write(dependent_qmd0_pointer=qmd_addr >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)
    self.qmds.append(qmd)
    self.prev_qmd = qmd

class NVCopyQueue(NVQueue):
  def copy(self, call:UOp):
    dest, src = (a.getaddr(self.devs) for a in call.src[1:3])
    for off in range(0, sz:=call.src[2].max_numel() * call.src[2].dtype.itemsize, step:=(1 << 31)):
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *hilo(src + UOp.const(off, dtypes.uint64)), *hilo(dest + UOp.const(off, dtypes.uint64)))
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, min(sz - off, step))
      self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA,
               nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"))

  def semaphore(self, addr:UOp, value:UOp, typ:str): # a one word release writes just the payload, a four word one the timestamp after it
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *hilo(addr), value.ccast(dtypes.uint32))
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type=f"release_{typ}_word_semaphore"))
  def timestamp(self, signal:UOp): self.semaphore(signal.getaddr(self.devs), UOp.const(0, dtypes.uint32), "four")
  def signal(self, signal:UOp, value:UOp): self.semaphore(signal.getaddr(self.devs), value, "one")

# *****************
# programs

class NVProgramData:
  def __init__(self, dev:NVDevice, obj:TinyELF):
    name, signature, mock = obj.name, obj.signature, isinstance(dev.iface, MOCKIface)
    self.constbufs: dict[int, tuple[int, int]] = {0: (0, 0x160)} # dict[constbuf index, tuple[offset in the image, size]]
    self.relocs: list[tuple[int, int, DType, int]] = [] # (byte offset in the image, symbol offset, width, shift) of the program's address
    self.prog_off, self.cbuf_0, sections, relocs = 0, [], list[Any](), list[Any]()
    image:bytes = obj.lib

    if (NAK:=isinstance(dev.renderer, NAKRenderer)):
      image = obj.lib[ctypes.sizeof(info:=mesa.struct_nak_shader_info.from_buffer_copy(obj.lib)):]
      regs, shmem, lcmem = info.num_gprs, round_up(info.cs.smem_size, 128), round_up(info.slm_size, 16)
    elif mock: image = obj.lib.ljust(round_up(len(obj.lib), 4), b'\x00') # for MOCKGPU the lib is PTX code, not an elf
    else:
      img, sections, relocs = elf_loader(obj.lib, force_section_align=128)
      image = bytes(img)
    prog_sz = len(image)

    if not NAK:
      # For MOCKGPU, the lib is PTX code, so some values are emulated.
      regs, shmem, lcmem, cbuf0_size = 0, 0x400, 0x240, 0x160 if mock else 0
      for sh in sections:
        if sh.name == f".nv.shared.{name}": shmem = round_up(0x400 + sh.header.sh_size, 128)
        if sh.name == f".text.{name}": self.prog_off, prog_sz = sh.header.sh_addr, sh.header.sh_size
        elif m:=re.match(r'\.nv\.constant(\d+)', sh.name): self.constbufs[int(m.group(1))] = (sh.header.sh_addr, sh.header.sh_size)
        elif sh.name.startswith(".nv.info"):
          for typ, param, data in self._parse_elf_info(sh):
            if sh.name == f".nv.info.{obj.name}" and param == 0xa: cbuf0_size = struct.unpack_from("IH", data)[1] # EIATTR_PARAM_CBANK
            elif sh.name == ".nv.info" and param == 0x12: lcmem = struct.unpack_from("II", data)[1] + 0x240 # EIATTR_MIN_STACK_SIZE
            elif sh.name == ".nv.info" and param == 0x2f: regs = struct.unpack_from("II", data)[1] # EIATTR_REGCOUNT

      # These reloc types are CUDA-specific: they all want the program's own address, which is only known once the linear links.
      for apply_image_offset, rel_sym_offset, typ, _ in relocs:
        if typ == 2: self.relocs.append((apply_image_offset, rel_sym_offset, dtypes.uint64, 0)) # R_CUDA_64
        elif typ == 0x38: self.relocs.append((apply_image_offset + 4, rel_sym_offset, dtypes.uint32, 0))
        elif typ == 0x39: self.relocs.append((apply_image_offset + 4, rel_sym_offset, dtypes.uint32, 32))
        else: raise RuntimeError(f"unknown NV reloc {typ}")

      # Minimum cbuf_0 size for driver params: Blackwell needs index 223 (224 entries), older GPUs need index 11 (12 entries)
      min_cbuf0_entries = 224 if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 12
      self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)

    # the arguments follow the driver params in constant buffer 0: the buffers as 64 bit addresses, then the vars packed by their width
    nbufs = sum(name is None for name, *_ in signature)
    self.vars = [dtypes.uint64 if mock else dt for _,_,dt,_ in signature[nbufs:]] # mockgpu wants every var 64 bit
    if mock: self.cbuf_0[80:82] = [nbufs, len(self.vars)] # mockgpu reads the arg counts out of cbuf0

    # NOTE: Ensure at least 4KB of space after the program to mitigate prefetch memory faults.
    self.image = image.ljust(round_up(len(image), 0x1000) + 0x1000, b'\x00')
    # constant buffer 0 holds the driver params and every argument after them, and starts 256 aligned like all constant buffers
    self.kernargs_size = round_up(max(self.constbufs[0][1], len(self.cbuf_0) * 4 + len(signature) * 8), 256)

    # Ensure device has enough local memory to run the program
    dev._ensure_has_local_memory(lcmem)

    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      if not NAK: self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(dev.shared_mem_window), *data64_le(dev.local_mem_window)], 0xfffdc0
      qmd = {'qmd_major_version':5, 'qmd_type':nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, 'register_count':regs,
        'shared_memory_size_shifted7':shmem>>7, f'shader_local_memory_{"low" if NAK else "high"}_size_shifted4':dev.slm_per_thread>>4}
    else:
      if not NAK: self.cbuf_0[6:12] = [*data64_le(dev.shared_mem_window), *data64_le(dev.local_mem_window), *data64_le(0xfffdc0)]
      qmd = {'qmd_major_version':3, 'sm_global_caching_enable':1, 'shared_memory_size':shmem, 'register_count_v':regs,
        f'shader_local_memory_{"low" if NAK else "high"}_size':dev.slm_per_thread}

    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= shmem) // 4096 + 1

    # the program and constant buffer addresses are patched into a copy of this at exec, everything else is the same for every launch
    self.qmd = QMD(dev)
    self.qmd.write(**qmd, qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalidate_0=1, min_sm_config_shared_mem_size=smem_cfg,
      target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a, program_prefetch_size=min(prog_sz>>8, 0x1ff),
      sass_version=dev.sass_version)
    for i,(_,sz) in self.constbufs.items(): self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})

    # Registers allocation granularity per warp is 256, warp allocation granularity is 4. Register file size is 65536.
    self.max_threads = ((65536 // round_up(max(1, regs) * 32, 256)) // 4) * 4 * 32

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off+4:start_off+sz+4] if typ == 0x4 else sz
      start_off += (sz if typ == 0x4 else 0) + 4

_nv_program_cache:dict[tuple[bytes, tuple[str, ...]], tuple[NVProgramData, UOp]] = {}
def nv_build_program(dev:NVDevice, prg:UOp, devs:tuple[str, ...]) -> tuple[NVProgramData, UOp]:
  if (cached:=_nv_program_cache.get(key:=(prg.src[3].arg, devs))) is None:
    data = NVProgramData(dev, prg.to_elf())
    buf = UOp.placeholder((len(data.image),), dtypes.uint8, next(UOp.unique_num), device=devs).rtag("program")
    rows = [(off, ((buf.getaddr(devs) + sym) >> sh).ccast(dt)) for off, sym, dt, sh in data.relocs]
    cached = _nv_program_cache[key] = (data, patch(buf, rows, data.image))
  return cached

class NVAllocator(Allocator['NVDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    return self.dev.iface.alloc(size, cpu_access=options.cpu_access, host=options.host, zero=options.zero)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    self.dev.iface.free(storage)
  def _map(self, buf:Buffer) -> BufferStorage: return self.dev.iface.map(buf)
  def _unmap(self, mapping:BufferStorage): self.dev.iface.unmap(mapping)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

  def _encode_decode(self, bufout:int, bufin:int, desc_buf:int, hist:list[int], shape:tuple[int,...], frame_pos:int):
    assert all(h % 0x100 == 0 for h in hist + [bufin, bufout, desc_buf]), "all buffers must be 0x100 aligned"

    h, w = ((2 * shape[0]) // 3 if shape[0] % 3 == 0 else (2 * shape[0] - 1) // 3), shape[1]
    dev, chroma_off = self.dev, round_up(w, 64) * round_up(h, 64)
    dev._ensure_has_vid_hw(w, h)

    cmds = nvm(4, nv_gpu.NVC9B0_SET_APPLICATION_ID, nv_gpu.NVC9B0_SET_APPLICATION_ID_ID_HEVC)
    cmds += nvm(4, nv_gpu.NVC9B0_SET_CONTROL_PARAMS, nv_flags("NVC9B0_SET_CONTROL_PARAMS", codec_type="hevc", testrun_env="prod_run", gptimer_on=1,
                err_conceal_on=1, mbtimer_on=1, event_trace_logging_on=1))
    cmds += nvm(4, nv_gpu.NVC9B0_SET_DRV_PIC_SETUP_OFFSET, desc_buf >> 8)
    cmds += nvm(4, nv_gpu.NVC9B0_SET_IN_BUF_BASE_OFFSET, bufin >> 8)
    for pos, buf in zip([(frame_pos-x) % (len(hist) + 1) for x in range(len(hist), 0, -1)] + [frame_pos], hist + [bufout]):
      cmds += nvm(4, nv_gpu.NVC9B0_SET_PICTURE_LUMA_OFFSET0 + pos*4, buf >> 8)
      cmds += nvm(4, nv_gpu.NVC9B0_SET_PICTURE_CHROMA_OFFSET0 + pos*4, (buf + chroma_off) >> 8)
    cmds += nvm(4, nv_gpu.NVC9B0_SET_COLOC_DATA_OFFSET, dev.vid_coloc_buf._buf >> 8)
    cmds += nvm(4, nv_gpu.NVC9B0_SET_NVDEC_STATUS_OFFSET, dev.vid_stat_buf._buf >> 8)
    cmds += nvm(4, nv_gpu.NVC9B0_HEVC_SET_TILE_SIZES_OFFSET, (desc_buf + 0x200) >> 8)
    cmds += nvm(4, nv_gpu.NVC9B0_HEVC_SET_FILTER_BUFFER_OFFSET, (filter_addr:=dev.vid_filter_buf._buf) >> 8)
    cmds += nvm(4, nv_gpu.NVC9B0_SET_INTRA_TOP_BUF_OFFSET, (filter_addr + dev.intra_top_off) >> 8)
    if dev.intra_unk_off is not None: cmds += nvm(4, 0x4dc, (filter_addr + dev.intra_unk_off) >> 8)
    cmds += nvm(4, nv_gpu.NVC9B0_EXECUTE, 0)
    dev._submit_cmds("NVDEC:0", *cmds)

# *****************
# device

@dataclass
class GPFifo: ring: Buffer; gpput: Buffer; doorbell: Buffer; put_value: Buffer; notifier: Buffer; entries: int; token: int # noqa: E702

class NVKIface:
  root = None
  fd_ctl: FileIOInterface
  fd_uvm: FileIOInterface
  count: int
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
      if int(drvver.driverVersionBuffer.decode().split('.')[0], 10) >= 610: nv_gpu = nv_610
      elif int(drvver.driverVersionBuffer.decode().split('.')[0], 10) >= 580: nv_gpu = nv_580

      self.uvm(nv_gpu.UVM_INITIALIZE, nv_gpu.UVM_INITIALIZE_PARAMS())

      # this error is okay, CUDA hits it too
      with contextlib.suppress(RuntimeError): self.uvm(nv_gpu.UVM_MM_INITIALIZE, nv_gpu.UVM_MM_INITIALIZE_PARAMS(uvmFd=self.fd_uvm.fd), self.fd_uvm_2)

      nv_iowr(NVKIface.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, gpus_info:=(nv_gpu.nv_ioctl_card_info_t*64)())
      NVKIface.gpus_info = hcq_filter_visible_devices([gi for gi in gpus_info if gi.valid], "NV")
      NVKIface.count = len(NVKIface.gpus_info)

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

  def rm_control(self, obj, cmd, params=None, **kwargs):
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

    for dev in [d for x in Device._opened_devices if isinstance(d:=Device[x], NVDevice) and not d.is_nvd()]:
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

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, map_flags=0, cpu_addr=None, **kwargs) -> BufferStorage:
    # Uncached memory is "system". Use huge pages only for gpu memory.
    page_size = mmap.PAGESIZE if uncached or host else ((2 << 20) if size >= (8 << 20) else (mmap.PAGESIZE if isinstance(self, MOCKIface) else
                                                                                             4 << 10))
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

  def free(self, storage:BufferStorage):
    mem = storage.meta
    if mem.hMemory > NVKIface.host_object_enumerator: # not a host object, clear phys mem.
      made = nv_gpu.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.dev.nvdevice, hObjectOld=mem.hMemory)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_FREE, made)
      if made.status != 0: raise RuntimeError(f"_gpu_free returned {get_error_str(made.status)}")
    self.uvm(nv_gpu.UVM_FREE, nv_gpu.UVM_FREE_PARAMS(base=storage.buf, length=mem.length))
    if storage.host is not None: FileIOInterface.munmap(storage.buf, mem.length)

  def unmap(self, mapping:BufferStorage):
    mem, owns_range = mapping.meta
    if owns_range: self.uvm(nv_gpu.UVM_FREE, nv_gpu.UVM_FREE_PARAMS(base=mapping.buf, length=mem.length))

  def _gpu_uvm_map(self, va_base, size, mem_handle, create_range=True, has_cpu_mapping=False) -> BufferStorage:
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
    return BufferStorage(va_base, uvm_map, MMIOInterface(va_base, size, fmt='B') if has_cpu_mapping else None)

  def map(self, buf:Buffer) -> BufferStorage:
    mem = buf.meta
    if buf.device.split(":")[0] in {"CPU", "PYTHON", "NPY"}:
      if buf._buf % 0x1000: raise RuntimeError("Host mapping requires a page-aligned address")
      if (mem:=next((m.meta[0] for d, m in buf.get_storage().maps.items() if d.device.startswith("NV")), None)) is None:
        return replace(mem:=self.alloc(buf.nbytes, host=True, cpu_addr=buf._buf), meta=(mem.meta, True))
    elif buf.device.split(":")[0] != "NV": raise RuntimeError(f"Cannot map {buf.device} on {self.dev.device}")
    return replace(mapping:=self._gpu_uvm_map(buf._buf, mem.length, mem.hMemory, create_range=False), meta=(mapping.meta, False))

  def _alloc_gpu_vaddr(self, size, alignment=(4 << 10), force_low=False):
    return NVKIface.low_uvm_vaddr_allocator.alloc(size, alignment) if force_low else NVKIface.uvm_vaddr_allocator.alloc(size, alignment)

  def sleep(self, tm:int): pass

class PCIIface(PCIIfaceBase):
  def __init__(self, dev, dev_id):
    # PCIIface's MAP_FIXED mmap will overwrite UVM allocations made by NVKIface, so don't try PCIIface if kernel driver was already used.
    if NVKIface.root is not None: raise RuntimeError("Cannot use PCIIface after NVKIface has been initialized (would corrupt UVM memory)")
    super().__init__(dev, dev_id, vendor=0x10de, devices=((0xff00, (0x2200,0x2400,0x2500,0x2600,0x2700,0x2800,0x2b00,0x2c00,0x2d00,0x2f00)),),
      base_class=0x03, vram_bar=1, va_start=NVMemoryManager.va_allocator.base, va_size=NVMemoryManager.va_allocator.size, dev_impl_t=NVDev)

    self.root, self.gpu_instance = 0xc1000000, 0
    self.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())

    # Setup classes for the GPU
    self.gpfifo_class, self.compute_class, self.dma_class = (gsp:=self.dev_impl.gsp).gpfifo_class, gsp.compute_class, gsp.dma_class
    self.viddec_class = gsp.viddec_class

  def setup_usermode(self): return 0xce000000, self.pci_dev.map_bar(bar=0, fmt='I', off=0xbb0000, size=0x10000)
  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass

  def rm_alloc(self, parent, clss, params=None, root=None) -> int: return self.dev_impl.gsp.rpc_rm_alloc(parent, clss, params, self.root)
  def rm_control(self, obj, cmd, params=None, **kwargs): return self.dev_impl.gsp.rpc_rm_control(obj, cmd, params, self.root, **kwargs)

  def device_fini(self): self.dev_impl.fini()

  def sleep(self, timeout):
    for _ in self.dev_impl.gsp.stat_q.read_resp(): pass
    if self.dev_impl.is_err_state: raise RuntimeError("Device fault detected")

class MOCKIface(NVKIface): count = 1

class NVDevice(Compiled):
  ifaces = [NVKIface, PCIIface, MOCKIface]
  sleep_timeout_ms = 200
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_nv_compute", name="submit"), lambda ctx, submit: encode_submit(NVComputeQueue(ctx, submit))),
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_nv_copy", name="submit"), lambda ctx, submit: encode_submit(NVCopyQueue(ctx, submit))),
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_nv_raw", name="submit"), lambda ctx, submit: encode_submit(NVQueue(ctx, submit))),
  ])

  def is_nvd(self) -> bool: return isinstance(self.iface, PCIIface)

  def __init__(self, device:str=""):
    self.iface = self._select_iface(device)

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
    self.vaspace = vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A, vaspace_params)

    self.iface.setup_vm(vaspace)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    self.channel_group = self.iface.rm_alloc(self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, channel_params)

    self.ctxshare = self.iface.rm_alloc(self.channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A,
      nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC))

    self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version = self._query_gpu_info('num_gpcs',
      'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version')

    # FIXME: no idea how to convert this for blackwells
    self.arch: str = "sm_120" if self.sm_version==0xa04 else f"sm_{(self.sm_version>>8)&0xff}{(val>>4) if (val:=self.sm_version&0xff) > 0xf else val}"
    self.sass_version = ((self.sm_version & 0xf00) >> 4) | (self.sm_version & 0xf)

    self.slm_per_thread = 0
    self.shader_local_mem:Buffer|None = None
    # Set windows addresses to not collide with other allocated buffers.
    self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000

    super().__init__(device, NVAllocator(self), [CUDARenderer, PTXRenderer, NVCCRenderer, NAKRenderer], None, arch=self.arch)

    self.pma_enabled, self.pma_exec_counter = PMA.value > 0 and PROFILE >= 1, itertools.count(0)

  @functools.cached_property
  def fifos(self) -> dict[str, GPFifo]:
    mem = self.iface.alloc(3<<20, contiguous=True, cpu_access=True, force_devmem=True, map_flags=nv_gpu.NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED<<23)
    self.gpfifo_buf = Buffer(self.device, 3<<20, dtypes.uint8, opaque=mem)

    compute = self._new_gpu_fifo("COMPUTE:0", self.ctxshare, self.channel_group, offset=0, entries=0x10000, compute=True)
    copy = self._new_gpu_fifo("COPY:0", self.ctxshare, self.channel_group, offset=0x100000, entries=0x10000)
    self.iface.rm_control(self.channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
    self.fifos = {"COMPUTE:0": compute, "COPY:0": copy}

    self._submit_cmds("COMPUTE:0", *nvm(1, nv_gpu.NVC6C0_SET_OBJECT, self.iface.compute_class),
                       *nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(self.local_mem_window)),
                       *nvm(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(self.shared_mem_window)))
    self._submit_cmds("COPY:0", *nvm(4, nv_gpu.NVC6C0_SET_OBJECT, self.iface.dma_class))

    if self.pma_enabled: self._prof_init() # the sampler binds to the channel group, so it only comes up once the channels do
    return self.fifos

  def _new_gpu_fifo(self, name:str, ctxshare, channel_group, offset=0, entries=0x400, compute=False, video=False) -> GPFifo:
    notifier = Buffer(self.device, size:=48 << 20, dtypes.uint8, opaque=self.iface.alloc(size, uncached=True))
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=self.gpfifo_buf._buf+offset, gpFifoEntries=entries,
      hObjectError=notifier.meta.hMemory, hObjectBuffer=self.virtmem if video else self.gpfifo_buf.meta.hMemory,
      hUserdMemory=(ctypes.c_uint32*8)(self.gpfifo_buf.meta.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset),
      engineType=19 if video else 0, hContextShare=ctxshare,
      hVASpace=self.vaspace if video and self.is_nvd() else 0) # gsp has no default vaspace, rm maps the decoder ctx into its own
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

    gpput_off = offset + entries*8 + getattr(nv_gpu.AmpereAControlGPFifo, 'GPPut').offset
    fifo = GPFifo(ring=self.gpfifo_buf.view(entries, dtypes.uint64, offset).ensure_allocated(),
      gpput=self.gpfifo_buf.view(1, dtypes.uint32, gpput_off).ensure_allocated(),
      doorbell=Buffer("CPU", 1, dtypes.uint32, options=BufferSpec(external_ptr=self.gpu_mmio.addr + 0x90), preallocate=True),
      put_value=Buffer("CPU", 1, dtypes.uint64, initial_value=bytes(8)), notifier=notifier, entries=entries, token=ws_token_params.workSubmitToken)
    self.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, tag=to_name(n, name)), lambda ctx, b=getattr(fifo, n): b)
                                        for n in ("ring", "gpput", "doorbell", "put_value")]) + self.pm_bufferize
    return fifo

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

  def _submit_cmds(self, queue:str, *cmds:int): # channel setup and video decode use the same runtime submit as kernels
    tl = timeline(devs:=(self.device,))
    value = tl.index(1).load()
    submit = make_submit(
      UOp(Ops.INS, arg=("wait", dtypes.void), src=(tl, value)),
      UOp(Ops.INS, arg=("nv", dtypes.void), src=(UOp(Ops.BINARY, arg=array.array('I', cmds).tobytes()),)),
      UOp(Ops.INS, arg=("store", dtypes.void), src=(tl, value + 1)), devs=devs, queue=queue).replace(arg="submit_nv_raw")
    call = UOp.sink(tl.after(submit).index(1).store(value + 1), arg=KernelInfo("nv_submit")).call(aux=HCQInfo(devs))
    linear = lower_and_compile(UOp(Ops.LINEAR, src=(unwrap(lower_call(call)),)))
    run_linear(hcq_link(linear, allow_cache=True), jit=True, update_stats=False, wait=True)

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required: return

    self.slm_per_thread = round_up(required, 32)
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    self.shader_local_mem = Buffer(self.device, round_up(bytes_per_tpc*self.num_tpc_per_gpc*self.num_gpcs, 0x20000), dtypes.uint8,
                                   options=BufferSpec(nolru=True), preallocate=True)

    self._submit_cmds("COMPUTE:0", *nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(self.shader_local_mem._buf)),
                       *nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(bytes_per_tpc), 0xff))

  def _ensure_has_vid_hw(self, w, h):
    if self.iface.viddec_class is None: raise RuntimeError(f"{self.device} Video decoder class not available.")

    coloc_sz = round_up((round_up(h, 64) * round_up(h, 64)) + (round_up(w, 64) * round_up(h, 64) // 16), 2 << 20)
    self.intra_top_off = round_up(h, 64) * (608 + 4864 + 152 + 2000)
    intra_unk_size = ((2 << 20) if self.iface.viddec_class >= nv_gpu.NVCFB0_VIDEO_DECODER else 0)
    self.intra_unk_off = (round_up(self.intra_top_off, 0x10000) + (64 << 10)) if intra_unk_size > 0 else None
    filter_sz = round_up(round_up(self.intra_top_off, 0x10000) + (64 << 10) + intra_unk_size, 2 << 20)

    def _vid_buf(sz): return Buffer(self.device, sz, dtypes.uint8, options=BufferSpec(zero=True, nolru=True), preallocate=True)
    if "NVDEC:0" not in self.fifos:
      self.fifos["NVDEC:0"] = self._new_gpu_fifo("NVDEC:0", 0, self.nvdevice, offset=0x200000, entries=2048, video=True)
      self.vid_coloc_buf, self.vid_filter_buf, self.vid_stat_buf = _vid_buf(coloc_sz), _vid_buf(filter_sz), _vid_buf(0x1000)
      self._submit_cmds("NVDEC:0", *nvm(4, nv_gpu.NVC6C0_SET_OBJECT, self.iface.viddec_class))
    else:
      if coloc_sz > self.vid_coloc_buf.nbytes: self.vid_coloc_buf = _vid_buf(coloc_sz)
      if filter_sz > self.vid_filter_buf.nbytes: self.vid_filter_buf = _vid_buf(filter_sz)

  def collect_prof(self):
    # the pc samples of a whole batch come back as one stream, so they are reported against the first kernel of it
    if self.pma_enabled and (ents:=list(self.prof_ents.values())) and (blob:=self._prof_readback()) is not None:
      Compiled.profile_events.append(ProfilePMAEvent(self.device, str(ents[0].name), blob, next(self.pma_exec_counter), ents[0].profile_key))
    super().collect_prof()

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
    self.profiler = self.iface.rm_alloc(self.subdevice, nv_gpu.MAXWELL_PROFILER_DEVICE,
      nv_gpu.NVB2CC_ALLOC_PARAMETERS(hClientTarget=self.iface.root, hContextTarget=self.channel_group))

    power_params = nv_gpu.struct_NVB0CC_CTRL_POWER_REQUEST_FEATURES_PARAMS(controlMask=(nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_ELCG_DISABLE << 0) | \
      (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_BLCG_DISABLE << 2) | (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_ELPG_DISABLE << 6) | \
      (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_IDLE_SLOWDOWN_DISABLE << 8) | (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_VAT_DISABLE << 10))
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES, power_params)

    self.pma_buf = Buffer(self.device, size:=getenv("PMA_BUFFER_SIZE", 512) << 20, dtypes.uint8,
                          opaque=self.iface.alloc(size, host=True, uncached=True, cpu_cached=True, cpu_access=True))
    self.pma_bytes = Buffer(self.device, size:=0x1000, dtypes.uint8,
                            opaque=self.iface.alloc(size, uncached=True, cpu_cached=True, cpu_access=self.is_nvd(), read_only=True))
    self.pma_rptr = 0

    pma_stream = nv_gpu.struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS(hMemPmaBuffer=self.pma_buf.meta.hMemory,
      pmaBufferSize=self.pma_buf.nbytes, hMemPmaBytesAvailable=self.pma_bytes.meta.hMemory, pmaBufferVA=self.pma_buf._buf)
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM, pma_stream, extra=(self.pma_buf, self.pma_bytes))

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

    start, end, view = self.pma_rptr, self.pma_rptr + params.bytesAvailable, self.pma_buf.host
    pma_data = bytes(view[start:min(end, self.pma_buf.nbytes)]) + bytes(view[:max(0, end - self.pma_buf.nbytes)])
    self.pma_rptr = end % self.pma_buf.nbytes

    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT,
      nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS(bytesConsumed=params.bytesAvailable))
    return pma_data

  def device_props(self) -> dict[str, Any]: return {'arch': self.arch, 'sm_version': self.sm_version}
