from __future__ import annotations
import subprocess, pathlib, struct, ctypes, tempfile, functools, platform, weakref, threading, array, sys
from tinygrad.helpers import to_mv, round_up, cache_dir, unwrap, prod, dedup, to_tuple
import tinygrad.runtime.support.objc as objc
from tinygrad.device import Buffer, BufferStorage, BufferSpec, Allocator, Compiled, Compiler, CompileError, MMIOInterface
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.runtime.autogen import metal
from tinygrad.runtime.support.c import DLL
from tinygrad.runtime.support.hcq2 import HWQueue, ccall, patch, layout_args, to_name
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, uopfunc
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops

# 13 is requestType that metal uses to compile source code into MTLB, there aren't any docs or symbols.
REQUEST_TYPE_COMPILE = 13

# Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
DLL("CoreGraphics", "CoreGraphics")

# FIXME: these need autogen to support objc categories
# https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjectiveC/Chapters/ocCategories.html
@functools.cache
def to_ns_str(s:str): return ctypes.cast(objc.msg("stringWithUTF8String:")(metal.NSString._objc_class_, s.encode()), metal.NSString).own()
def checked(fn, *args): # fn(*args, &error), raised if set
  ret = fn(*args, ctypes.byref(err:=metal.NSError()))
  if err.value is not None: raise RuntimeError(bytes(objc.msg("UTF8String", ctypes.c_char_p)(err.localizedDescription())).decode())
  return ret

class MetalCompiler(Compiler):
  # Opening METAL after LLVM doesn't fail because ctypes.CDLL opens with RTLD_LOCAL but MTLCompiler opens it's own llvm with RTLD_GLOBAL
  # This means that MTLCompiler's llvm will create it's own instances of global state because RTLD_LOCAL doesn't export symbols, but if RTLD_GLOBAL
  # library is loaded first then RTLD_LOCAL library will just use it's symbols. On linux there is RTLD_DEEPBIND to prevent that, but on macos there
  # doesn't seem to be anything we can do.
  import tinygrad.runtime.autogen.llvm as _
  support = DLL("MTLCompiler", "MTLCompiler")
  support.MTLCodeGenServiceCreate.restype = ctypes.c_void_p

  def __init__(self):
    self.cgs = ctypes.c_void_p(MetalCompiler.support.MTLCodeGenServiceCreate(b"tinygrad"))
    super().__init__("compile_metal_direct")
  def __reduce__(self): return (MetalCompiler,()) # force pickle to create new instance for each multiprocessing fork
  def compile(self, src:str) -> bytes:
    ret: Exception|bytes = CompileError("MTLCodeGenServiceBuildRequest returned without calling the callback")
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
    def callback(blockptr, error, dataPtr, dataLen, errorMessage):
      nonlocal ret
      if error == 0:
        reply = bytes(to_mv(dataPtr, dataLen))
        # offset from beginning to data = header size + warning size
        ret = reply[sum(struct.unpack('<LL', reply[8:16])):]
      else:
        ret = CompileError(errorMessage.decode())

    # no changes for compute in 2.0 - 2.4 specs, use 2.0 as default for old versions.
    macos_major = int(platform.mac_ver()[0].split('.')[0])
    metal_version = "metal4.0" if macos_major >= 26 else "metal3.1" if macos_major >= 14 else "metal3.0" if macos_major >= 13 else "macos-metal2.0"

    # llvm will create modules.timestamp in cache path and cache compilation of metal stdlib (250ms => 8ms compilation time)
    # note that llvm won't necessarily create anything else here as apple has prebuilt versions of many standard libraries
    params = f'-fno-fast-math -std={metal_version} --driver-mode=metal -x metal -fmodules-cache-path="{cache_dir}" -fno-caret-diagnostics'
    # source blob has to be padded to multiple of 4 but at least one 'b\x00' should be added, params blob just has to be null terminated
    src_padded, params_padded = src.encode() + b'\x00'*(round_up(len(src) + 1, 4) - len(src)), params.encode() + b'\x00'
    request = struct.pack('<QQ', len(src_padded), len(params_padded)) + src_padded + params_padded
    # The callback is actually not a callback but a block which is apple's non-standard extension to add closures to C.
    # See https://clang.llvm.org/docs/Block-ABI-Apple.html#high-level for struct layout.
    # Fields other than invoke are unused in this case so we can just use ctypes.byref with negative offset to invoke field, add blockptr as a first
    # argument and pretend it's a normal callback
    MetalCompiler.support.MTLCodeGenServiceBuildRequest(self.cgs, None, REQUEST_TYPE_COMPILE, request, len(request), ctypes.byref(callback, -0x10))
    if isinstance(ret, Exception): raise ret
    assert ret[:4] == b"MTLB" and ret[-4:] == b"ENDT", f"Invalid Metal library. {ret!r}"
    return ret
  def disassemble(self, lib:bytes):
    with tempfile.NamedTemporaryFile(delete=True) as shader:
      shader.write(lib)
      shader.flush()
      proc = subprocess.Popen(f"cd {pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}",
                              stdout=subprocess.PIPE, shell=True, text=True, bufsize=1)
      for line in unwrap(proc.stdout): print(line, end="")
      ret = proc.wait()
      if ret: print("Disassembler Error: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")

# *****************
# queue

HANDLES = ("queue", "event", "fence", "resources", "count")
MSGSEND, SELNAME = [metal.dll.bind(ctypes.c_void_p)(f) for f in (metal.dll.objc_msgSend, metal.dll.sel_registerName)]

# the slot of a handle of the device
def mtl_handle(d, name:str) -> UOp:
  return UOp.alloc((len(HANDLES),), dtypes.uint64, 0, device=to_tuple(d)[0]).rtag(to_name(to_tuple(d)[0], "handles"))[(i:=HANDLES.index(name)):i+1]

@uopfunc
def mtl_send(obj:UOp, sel:UOp, a:UOp, b:UOp, c:UOp, out:UOp|None=None) -> UOp: # objc_msgSend by the selector's name, the result to out
  c = ccall(MSGSEND, obj, ccall(SELNAME, sel), a, b, c)
  return (out.index(0).store(c) if out is not None else c).sink()

def mtl_msg(obj:UOp, sel:str, *args:UOp|int, out:UOp|None=None) -> UOp:
  words = [UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in (*args, 0, 0, 0)][:3] # the extra args are ignored
  return mtl_send(obj.index(0).load(), UOp(Ops.BINARY, arg=sel.encode() + b"\0").index(0), *words, out=out)

@uopfunc
def mtl_run(icb:UOp, value:UOp, first:UOp|int, count:int, last:bool, q:MetalQueue, stamp:UOp|None=None) -> UOp:
  cmds, hdr, devs, dev = q.cmds, round_up(q.nbytes, 8) + 24, q.devs, q.dev # a command buffer for the commands [first, first + count)
  cb, enc = [UOp.alloc((1,), dtypes.uint64, 0, device=devs[0]).rtag(t) for t in ("mtl_cb", "mtl_enc")]
  fence, event = [mtl_handle(devs, h).index(0).load() for h in ("fence", "event")]
  c = mtl_msg(mtl_handle(devs, "queue"), "commandBuffer", out=cb)
  c = mtl_msg(cb.after(c), "computeCommandEncoder", out=enc)
  c = mtl_msg(enc.after(c), "waitForFence:", fence)
  if dev.residency.value is None: # no residency set: declare the buffers
    c = mtl_msg(enc.after(c), "useResources:count:usage:", *[mtl_handle(devs, h).index(0).load() for h in ("resources", "count")], 3)

  # before apple9 the encoder must use the pipelines
  if not dev.arch.startswith("Apple") or int(dev.arch[5:]) < 9:
    r = UOp.range(len(dedup(c[:2] for c in cmds)), next(UOp.unique_num), dtype=dtypes.uint64)
    c = mtl_msg(enc.after(c), "setComputePipelineState:", icb.bitcast(dtypes.uint64).index(hdr // 8 + 1 + len(cmds) + r).load())
    c = mtl_msg(enc.after(c), "dispatchThreadgroups:threadsPerThreadgroup:", icb.index(hdr - 24), icb.index(hdr - 24)).end(r)

  c = mtl_msg(enc.after(c), "executeCommandsInBuffer:withRange:", icb.bitcast(dtypes.uint64).index(hdr // 8).load(), first, count)
  c = mtl_msg(enc.after(c), "updateFence:", fence)
  c = mtl_msg(enc.after(c), "endEncoding")

  if stamp is not None: # write meta to collect timestamps: [command buffer, 0] until synchronize reads its times. MTL4 solves that dance
    c = stamp.after(c).index(3).store(0)
    c = stamp.after(c).index(1).store(cb.after(c).index(0).load())
  if last: c = mtl_msg(cb.after(c), "encodeSignalEvent:value:", event, value)
  return mtl_msg(cb.after(c), "commit").sink()

class MetalQueue(HWQueue):
  dev:MetalDevice
  def __init__(self, submit:UOp):
    super().__init__(submit)
    self.rows, self.cmds, self.sizes, self.stamps, self.nbytes = list[tuple[int, UOp]](), list[tuple](), list[tuple[int, int]](), list[UOp](), 0

  def exec(self, call:UOp, prg:UOp):
    bufs, vals, obj = get_call_arg_uops(call), get_call_var_uops(call, prg), prg.to_elf()
    args = [bufs[i].getaddr(self.devs) for i in prg.arg.globals] + [v.ccast(var.dtype) for v, var in zip(vals, prg.arg.vars)]
    self.rows += (rows:=layout_args(args, off:=round_up(self.nbytes, 256)))
    self.nbytes = max([o + w.dtype.itemsize for o, w in rows], default=off + 8)

    # symbolic sizes, set on the command at run time
    dims = (*prg.arg.global_size, *prg.arg.local_size)
    if any(isinstance(d, UOp) for d in dims):
      self.sizes.append((len(self.cmds), at:=round_up(self.nbytes, 8)))
      self.rows += layout_args([d.cast(dtypes.uint64) if isinstance(d, UOp) else UOp.const(d, dtypes.uint64) for d in dims], at)
      self.nbytes = at + 48
    self.cmds.append((obj.lib, obj.name, tuple(1 if isinstance(d, UOp) else int(d) for d in dims), off))

  def wait(self, dst:UOp, val:UOp, eq=False): pass # the fence orders the queue
  def timestamp(self, dst:UOp): self.stamps.append(dst)
  def signal(self, dst:UOp, val:UOp): self.value = val

  def submit(self, cmdbuf:UOp) -> UOp:
    n, zero, pipes = len(self.cmds), round_up(self.nbytes, 8), dedup(c[:2] for c in self.cmds)
    tag = (self.dev.tag("mtl_icb"), tuple(self.cmds), zero + 24)
    buf = UOp.alloc((zero + 24 + 8 * (1 + n + len(pipes)),), dtypes.uint8, device=self.devs[0]).rtag(tag).after(*self.deps)
    icb = patch(buf, self.rows + [(zero + 8 * i, UOp.const(0, dtypes.uint64)) for i in range(3)])

    # symbolic sizes
    for ci, off in self.sizes:
      cmd = icb.bitcast(dtypes.uint64)[zero // 8 + 4 + ci:zero // 8 + 5 + ci]
      icb = icb.after(mtl_msg(cmd, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", icb.index(off), icb.index(off + 24)))

    # collect timestamps using cmdbuf metrics, so sep cmdbufs
    if not self.stamps: return mtl_run(icb, self.value, 0, n, True, self)
    slots, r = self.stamps[0].src[0], UOp.range(n - 1, next(UOp.unique_num), dtype=dtypes.uint64) # slots: [signal, timeline, [x, cb, x, end]...]
    if n > 1: icb = icb.after(mtl_run(icb.after(r), self.value, r, 1, False, self, slots.shrink(((4 + 4 * r, 8 + 4 * r),))).end(r))
    return mtl_run(icb, self.value, n - 1, 1, True, self, slots.shrink(((4 * n, 4 * n + 4),)))

# *****************
# device

class MetalAllocator(Allocator['MetalDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    mtl = metal.MTLBuffer(options.external_ptr) if options.external_ptr else \
          self.dev.sysdevice.newBufferWithLength_options(size, metal.MTLResourceStorageModeShared)
    if mtl.value is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    self.dev.mark_resident(mtl, True)
    return BufferStorage(mtl.gpuAddress(), mtl, MMIOInterface(c, size) if (c:=mtl.contents()) else None)
  def _free(self, storage:BufferStorage, options:BufferSpec): # metal doesn't track what a kernel reaches
    self.dev.synchronize()
    self.dev.mark_resident(storage.meta, False)
    storage.meta.retain = False
    storage.meta.release()
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

class MetalDevice(Compiled):
  has_copy_queue = False
  pm_encode = PatternMatcher([
    (UPat(Ops.CALL, src=(UPat.custom_function("submit_metal_compute"), UPat()), name="s"), lambda s: MetalQueue(s).encode()),
  ])
  pm_lower = PatternMatcher([
    (UPat(Ops.PARAM, name="p").f(Ops.AFTER, allow_any_len=True, name="t").index(UPat.const(0)).load(), lambda p, t: None if p.arg.name != "tl" else \
     (r:=UOp.placeholder((1,), dtypes.uint64, None, AddrSpace.REG)).after(mtl_msg(mtl_handle(p.device, "event").after(t), "signaledValue", out=r))[0])
  ])

  def __init__(self, device:str=""):
    self.sysdevice = metal.MTLCreateSystemDefaultDevice()

    # queue allocation
    self.queue = self.sysdevice.newCommandQueueWithMaxCommandBufferCount(1024)
    if self.queue.value is None: raise RuntimeError("Cannot allocate a new command queue")

    # try to use residency set when supported
    rsd = ctypes.cast(objc.msg("new", clsmeth=True)(metal.MTLResidencySetDescriptor), metal.MTLResidencySetDescriptor)
    self.residency = self.sysdevice.newResidencySetWithDescriptor_error(rsd, None)
    if self.residency.value is not None: self.queue.addResidencySet(self.residency)

    self.resources, self.table = list[int](), (ctypes.c_uint64 * 1)()
    self.event, self.fence = self.sysdevice.newSharedEvent(), self.sysdevice.newFence()
    self.icbs, self.profile_slots = weakref.WeakKeyDictionary[Buffer, tuple](), weakref.WeakSet[Buffer]()

    # https://developer.apple.com/documentation/metal/mtlgpufamily
    def check_family(f): return next(filter(self.sysdevice.supportsFamily, reversed([v for v, nm in metal.enum_MTLGPUFamily.items() if f in nm])), 0)
    super().__init__(device, MetalAllocator(self), [MetalRenderer], None,
                     arch=metal.enum_MTLGPUFamily[check_family("Apple") or check_family("Mac")][12:])
    Compiled.pm_bufferize += PatternMatcher([
      (UPat(Ops.ALLOC, tag=self.tag("handles")), lambda d=self: d.handles),
      (UPat(Ops.ALLOC, tag="slots", name="b"), # with stamps
       lambda b, d=self: d.new_slots(b.max_numel()) if b.device == d.device and b.max_numel() > 4 else None),
      (UPat(Ops.ALLOC, name="b"), lambda b, d=self: d.new_icb(*b.tag[1:]) if isinstance(b.tag, tuple) and b.tag[0] == d.tag("mtl_icb") else None)])

  @functools.cached_property
  def handles(self) -> Buffer:
    vals = [self.queue.value, self.event.value, self.fence.value, ctypes.addressof(self.table), len(self.resources)]
    return Buffer(self.host, len(vals) * 8, initial_value=struct.pack(f"{len(vals)}Q", *vals))

  def mark_resident(self, mtl:metal.MTLBuffer, add:bool):
    if self.residency.value is not None:
      objc.msg("addAllocation:" if add else "removeAllocation:", None, [objc.id_])(self.residency, mtl)
      return objc.msg("commit", None)(self.residency)

    self.resources.append(unwrap(mtl.value)) if add else self.resources.remove(unwrap(mtl.value))
    self.table = (ctypes.c_uint64 * max(len(self.resources), 1))(*self.resources)

    # update sels table
    if "handles" in self.__dict__: self.handles.host.view(fmt='Q')[3:5] = array.array('Q', [ctypes.addressof(self.table), len(self.resources)])

  def new_slots(self, n:int) -> Buffer:
    self.profile_slots.add(buf:=Buffer(self.host, n * 8, initial_value=bytes(8 * n)))
    return buf

  @functools.cache
  def pipeline(self, lib:bytes, name:str) -> metal.MTLComputePipelineState:
    library = checked(self.sysdevice.newLibraryWithData_error, objc.dispatch_data_create(lib, len(lib), None, None))
    descriptor = metal.MTLComputePipelineDescriptor.new()
    descriptor.setComputeFunction(library.newFunctionWithName(to_ns_str(name)))
    descriptor.setSupportIndirectCommandBuffers(True)
    return checked(self.sysdevice.newComputePipelineStateWithDescriptor_options_reflection_error, descriptor, metal.MTLPipelineOptionNone, None)

  def new_icb(self, cmds:tuple[tuple[bytes, str, tuple[int, ...], int], ...], header:int) -> Buffer:
    pipes = dedup(c[:2] for c in cmds)
    buf = Buffer(self.device, header + 8 * (1 + len(cmds) + len(pipes)), options=BufferSpec(nolru=True), preallocate=True)
    desc = metal.MTLIndirectCommandBufferDescriptor.new()
    desc.setCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
    desc.setMaxKernelBufferBindCount(1)
    icb = self.sysdevice.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(desc, max(len(cmds), 1), 0)
    if icb.value is None: raise RuntimeError("create indirect command buffer failed, does your system support this?")
    commands = [icb.indirectComputeCommandAtIndex(i).own() for i in range(len(cmds))]
    for cmd, (lib, name, dims, off) in zip(commands, cmds):
      cmd.setComputePipelineState(state:=self.pipeline(lib, name))
      if prod(dims[3:]) > (mx:=state.maxTotalThreadsPerThreadgroup()): raise RuntimeError(f"local size {dims[3:]} bigger than {mx}")
      cmd.setKernelBuffer_offset_atIndex(buf.get_storage().meta, off, 0)
      cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(metal.MTLSize(*dims[:3]), metal.MTLSize(*dims[3:]))
      cmd.setBarrier()
    buf.host.view(fmt='Q')[header // 8:] = array.array('Q', [icb.value, *[c.value for c in commands], *[self.pipeline(*p).value for p in pipes]])
    self.icbs[buf] = (icb, commands)
    return buf

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    if sys.is_finalizing(): return # the event doesn't wake at exit
    wait = objc.msg("waitUntilSignaledValue:timeoutMS:", ctypes.c_bool, [ctypes.c_uint64, ctypes.c_uint64])
    if not wait(self.event, value, int(self.wait_timeout_ms)): raise RuntimeError(f"{self.device} signal wait timed out")

  def synchronize(self, timeout:int|None=None):
    for buf in list(self.profile_slots): # pending: [command buffer, 0]
      slots = buf.host.view(fmt='Q')
      for start in range(5, buf.nbytes // 8, 4):
        if slots[start] and not slots[start + 2]:
          (cb:=metal.MTLCommandBuffer(slots[start])).waitUntilCompleted()
          slots[start], slots[start + 2] = int(cb.GPUStartTime() * 1e9), int(cb.GPUEndTime() * 1e9)

    super().synchronize(timeout)
    if (pool:=getattr(pools, "pool", None)) is not None: objc.lib.objc_autoreleasePoolPop(pool)
    pools.pool = objc.lib.objc_autoreleasePoolPush()

pools = threading.local()
