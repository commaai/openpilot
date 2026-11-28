from __future__ import annotations
import functools, os, glob, pathlib, struct, itertools
from typing import cast
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import ceildiv, round_up, to_tuple, flatten, wait_cond
from tinygrad.codegen.compile import get_call_arg_uops
from tinygrad.runtime.autogen import nvme, pci
from tinygrad.runtime.support.hcq2 import to_name, ins, unwrap_view, copy_submit
from tinygrad.runtime.support.memory import AddrSpace, MemoryManager, MMIOInterface
from tinygrad.runtime.support.system import PCIDevice, PCIIfaceBase, System, filter_visible_devices
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

NVME_IDS = (0x15b7, ((0xffff, (0x5030, 0x5050)),), 0x01) # vendor, (mask, device ids), base class
SQE, CQE, ENTRIES = 1 << nvme.NVME_NVM_IOSQES, 1 << nvme.NVME_NVM_IOCQES, 0x1000 >> nvme.NVME_NVM_IOSQES # a queue is one page, no iommu needed

class NVMEQueue:
  def __init__(self, dev:NVMEDev, qid:int):
    self.dev, self.qid, self.tail, self.phase = dev, qid, 0, 1
    (self.sq, (self.sq_addr,)), (self.cq, (self.cq_addr,)) = dev.pci_dev.alloc_sysmem(0x1000), dev.pci_dev.alloc_sysmem(0x1000)
    self.sq_db, self.cq_db = nvme.NVME_REG_DBS + 2 * qid * dev.db_stride, nvme.NVME_REG_DBS + (2 * qid + 1) * dev.db_stride

  def submit(self, opcode:int, nsid:int=0, prp1:int=0, cdw10:int=0, cdw11:int=0):
    slot, self.tail = self.tail, (self.tail + 1) % ENTRIES
    self.sq[slot * SQE:(slot + 1) * SQE] = struct.pack("<IIQQQQII16x", opcode | slot << 16, nsid, 0, 0, prp1, 0, cdw10, cdw11)
    System.memory_barrier()
    self.dev.regs[self.sq_db // 4] = self.tail
    wait_cond(lambda: self.cq.view(slot * CQE + 12, 4, fmt='I')[0] >> 16 & 1, value=self.phase, msg=f"nvme queue {self.qid}")
    if (dw3:=self.cq.view(slot * CQE + 12, 4, fmt='I')[0]) >> 17:
      raise RuntimeError(f"nvme {self.dev.pci_dev.pcibus}: queue {self.qid} command {opcode:#x} failed, status {dw3 >> 17:#x}")
    self.phase ^= int(self.tail == 0)
    self.dev.regs[self.cq_db // 4] = self.tail

class NVMEDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev = pci_dev
    pci_dev.write_config(pci.PCI_COMMAND, pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, 2)
    self.regs = pci_dev.map_bar(0, fmt='I')
    cap = self.regs[nvme.NVME_REG_CAP // 4] | self.regs[nvme.NVME_REG_CAP // 4 + 1] << 32
    self.db_stride, timeout_ms = 4 << (cap >> 32 & 0xf), (cap >> 24 & 0xff) * 500

    self.regs[nvme.NVME_REG_CC // 4] = 0
    wait_cond(lambda: self.regs[nvme.NVME_REG_CSTS // 4] & nvme.NVME_CSTS_RDY, value=0, timeout_ms=timeout_ms, msg="nvme disable")
    self.admin_q = NVMEQueue(self, 0)
    self.regs[nvme.NVME_REG_AQA // 4] = (ENTRIES - 1) << 16 | (ENTRIES - 1)
    for reg, addr in ((nvme.NVME_REG_ASQ, self.admin_q.sq_addr), (nvme.NVME_REG_ACQ, self.admin_q.cq_addr)):
      self.regs[reg // 4], self.regs[reg // 4 + 1] = addr & 0xffffffff, addr >> 32
    self.regs[nvme.NVME_REG_CC // 4] = nvme.NVME_CC_IOCQES | nvme.NVME_CC_IOSQES | nvme.NVME_CC_CSS_NVM | nvme.NVME_CC_ENABLE
    wait_cond(lambda: self.regs[nvme.NVME_REG_CSTS // 4] & nvme.NVME_CSTS_RDY, value=1, timeout_ms=timeout_ms, msg="nvme enable")

    ident, (ident_addr,) = pci_dev.alloc_sysmem(0x1000)
    self.admin_q.submit(nvme.nvme_admin_identify, 0, ident_addr, 1)
    self.max_xfer = 0x1000 << min(ident[77] or 8, 8) # at most prp1 and a 256 entry list
    self.admin_q.submit(nvme.nvme_admin_identify, 1, ident_addr, 0)
    self.lba_shift = ident[128 + 4 * (ident[26] & 0xf) + 2]
    self.size = struct.unpack_from("<Q", bytes(ident[0:8]))[0] << self.lba_shift
    self.admin_q.submit(nvme.nvme_admin_set_features, 0, 0, nvme.NVME_FEAT_NUM_QUEUES, 0xfffe << 16 | 0xfffe)
    self.queues:list[NVMEQueue] = []

  def new_queue(self) -> NVMEQueue:
    self.queues.append(q:=NVMEQueue(self, len(self.queues) + 1))
    self.admin_q.submit(nvme.nvme_admin_create_cq, 0, q.cq_addr, (ENTRIES - 1) << 16 | q.qid, 1)
    self.admin_q.submit(nvme.nvme_admin_create_sq, 0, q.sq_addr, (ENTRIES - 1) << 16 | q.qid, q.qid << 16 | 1)
    return q

def prp_table(dev:NVMEDev, pages:list[int]) -> tuple[MMIOInterface, list[int]]:
  lists, list_addrs = dev.pci_dev.alloc_sysmem(ceildiv(len(pages) + 1, 256) * 0x1000)
  for j in range(len(list_addrs)):
    chunk = pages[256 * j:256 * j + 512] # list pages overlap, so a prp list never crosses a page and needs no chaining
    lists[j * 0x1000:j * 0x1000 + 8 * len(chunk)] = struct.pack(f"<{len(chunk)}Q", *chunk)
  table, table_addrs = dev.pci_dev.alloc_sysmem(round_up(16 * len(pages), 0x1000))
  # each page and a pointer to the prp list of the pages after it
  entries = flatten((page, list_addrs[(p + 1) // 256] + (p + 1) % 256 * 8) for p, page in enumerate(pages))
  table[:8 * len(entries)] = struct.pack(f"<{len(entries)}Q", *entries)
  return table, table_addrs

class PCIIface(PCIIfaceBase):
  def __init__(self, dev:NVMEDevice, index:int):
    # make sure the drive isn't mounted or used by md/lvm
    pcibus = filter_visible_devices(System.list_devices(*NVME_IDS), "NVME")[index][1]
    mounted = {line.split()[0] for line in pathlib.Path("/proc/mounts").read_text().splitlines()}
    for b in [b for b in glob.glob("/sys/class/block/*") if f"/{pcibus}/" in os.path.realpath(b)]:
      assert f"/dev/{os.path.basename(b)}" not in mounted and not os.listdir(f"{b}/holders"), f"{dev.device} ({pcibus}) is in use by the system: {b}"
    super().__init__(dev, index, *NVME_IDS[:2], vram_bar=0, va_start=MemoryManager.va_allocator.base, va_size=MemoryManager.va_allocator.size,
                     dev_impl_t=NVMEDev, base_class=NVME_IDS[2])

class NVMEAllocator(Allocator['NVMEDevice']):
  lru = False
  def __init__(self, dev:NVMEDevice):
    super().__init__(dev)
    self.prp_tables:dict[int, Buffer] = {}
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: return BufferStorage(0)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset
  def _map(self, buf:Buffer) -> BufferStorage:
    iface = getattr(Device[buf.device], "iface", None)
    if not isinstance(iface, PCIIfaceBase) or iface.peer_group != self.dev.peer_group: raise RuntimeError(f"NVME can't reach {buf.device}")
    mapping = buf.meta.mapping
    paddrs = mapping.paddrs if mapping.aspace is AddrSpace.SYS else PCIIfaceBase.p2p_paddrs(iface, mapping.paddrs)[0]
    table = self.dev.iface.buffer(*prp_table(self.dev.nvme, [p + off for p, size in paddrs for off in range(0, size, 0x1000)]))
    # Root the table until _unmap; cyclic GC can finalize a Buffer held only by cached storage's metadata.
    self.prp_tables[addr:=table.get_buf(buf.device)] = table
    return BufferStorage(addr)
  def _unmap(self, storage:BufferStorage): self.prp_tables.pop(storage.buf).deallocate()

class NVMEDevice(Compiled):
  ifaces = [PCIIface]

  def __init__(self, device:str):
    self.iface = self._select_iface(device)
    self.nvme = cast(NVMEDev, self.iface.dev_impl)
    super().__init__(device, NVMEAllocator(self), None, props={"max_xfer": self.nvme.max_xfer, "lba_shift": self.nvme.lba_shift})
    Compiled.link_bufs |= {to_name(device, f"gpu{i}", n): lambda b, i=i, n=n: self.gpu_queue(i)[n] for i in range(16)
                           for n in ("sq", "cq", "seq", "sq_db", "cq_db")}
  def synchronize(self, timeout:int|None=None): pass

  @functools.cache
  def gpu_queue(self, gpu:int) -> dict[str, Buffer]:
    q, (seq, seq_addrs), bar0 = self.nvme.new_queue(), self.nvme.pci_dev.alloc_sysmem(0x1000), self.iface.pci_dev.bar_info(0)[0]
    regs = self.iface.buffer(self.nvme.regs, [bar0 + o for o in range(0, self.nvme.regs.nbytes, 0x1000)], snooped=False)
    return {"sq": self.iface.buffer(q.sq, [q.sq_addr]), "cq": self.iface.buffer(q.cq, [q.cq_addr]),
            "seq": self.iface.buffer(seq.view(0, 8), seq_addrs), "sq_db": regs.view(4, q.sq_db), "cq_db": regs.view(4, q.cq_db)}

# *****************
# UOps implementation

def nvme_copies(queue:UOp, calls:list[UOp]) -> list[list[tuple]]: # builds the commands for all reads from one drive on one gpu queue
  drive, gpu = to_tuple(get_call_arg_uops(calls[0])[1].device)[0], queue.device
  slot = f"gpu{gpu.split(':')[1] if ':' in gpu else 0}"
  max_xfer, lba_shift = (t:=Device.target_for(drive)).get("max_xfer", 128 << 10), t.get("lba_shift", 9)
  def mem(name:str, size:int, dtype=dtypes.uint8) -> UOp: return UOp.extern((size,), dtype, drive, to_name(slot, name))
  sq, cq = mem("sq", 0x1000).getaddr(gpu), mem("cq", 0x1000).getaddr(gpu)
  sq_db, cq_db = mem("sq_db", 4).getaddr(gpu), mem("cq_db", 4).getaddr(gpu)

  # where this submit starts in the queue, bumped once for all its commands
  cmds = [ceildiv(get_call_arg_uops(c)[1].nbytes(), max_xfer) for c in calls]
  start = (seq:=mem("seq", 1, dtypes.uint64)).after(seq.index(0).store(seq.index(0).load() + sum(cmds))).index(0).load() - sum(cmds)

  copies = []
  for call, off in zip(calls, itertools.accumulate(cmds, initial=0)):
    n = start + off
    (dst, src), (full, rem) = get_call_arg_uops(call), divmod(get_call_arg_uops(call)[1].nbytes(), max_xfer)
    (dbase, doff), soff = unwrap_view(dst), unwrap_view(src)[1]
    assert doff % 0x1000 == 0 and (soff | src.nbytes()) % (1 << lba_shift) == 0, \
      f"{drive} reads whole blocks of {1 << lba_shift} bytes into page aligned memory, not {src.nbytes()} bytes at {soff} to {doff}"
    prps, lba0 = dbase.getaddr(drive) + doff // 0x1000 * 16, src.getaddr(drive) >> lba_shift
    def done(end:UOp|int) -> list[UOp]: # completes the read's commands before end: waits for the last one's cqe, then rings the cq doorbell
      return [ins("wait", cq + ((n + end - 1) % ENTRIES) * CQE + 12, 1), ins("write", cq_db, ((n + end) % ENTRIES).cast(dtypes.uint32))]
    def issue(cmd:UOp|int, count:int) -> list[UOp]: # issues the read's commands cmd to cmd + count
      ops = []
      for j in range(count): # write the sqe, then copy prp1 and prp2 (the next page or a list) from the table
        c, slot = cmd + j, (n + cmd + j) % ENTRIES
        nb = rem if isinstance(c, int) and c == full else max_xfer
        entry, lba, page = sq + slot * SQE, lba0 + c * (max_xfer >> lba_shift), c * (max_xfer // 0x1000)
        cid = c % 0xffff + 1 # the cid is 16 bits: never 0, so the cqe reads as nonzero, and unique among the commands in flight
        ops += [ins("write", entry, UOp.const(cid << 16 | nvme.nvme_cmd_read, dtypes.uint32), 1, *[0] * 8, lba.cast(dtypes.uint32),
                    (lba >> 32).cast(dtypes.uint32), (nb >> lba_shift) - 1, 0, 0, 0),
                ins("copy", entry + 24, prps + page * 16, 16 if ceildiv(nb, 0x1000) > 2 else 8)] + \
               ([ins("copy", entry + 32, prps + (page + 1) * 16, 8)] if ceildiv(nb, 0x1000) == 2 else [])
      cqe = cq + ((n + cmd + count - 1) % ENTRIES) * CQE + 12 # the wave's last cqe, cleared before ringing the doorbell
      return ops + [ins("write", cqe, 0), ins("write", sq_db, ((n + cmd + count) % ENTRIES).cast(dtypes.uint32))]
    # waves of commands, each issued before the previous one is completed: two are in flight, so the drive always has one queued
    nwaves, left = full // (wave:=(ENTRIES - 1) // 2), full % wave + (rem > 0)
    first = [(issue(0, wave), list[UOp](), None)] if nwaves else []
    r = UOp.range(nwaves - 1, next(UOp.unique_num), dtype=dtypes.u64) # loops over the waves after the first
    middle = [(issue((r + 1) * wave, wave), done((r + 1) * wave), r)] if nwaves > 1 else []
    last = [(issue(nwaves * wave, left), done(nwaves * wave) if nwaves else [], None)] if left else []
    copies.append(first + middle + last + [([], (done(full + (rem > 0)) if full or rem else []) + [ins("memory_barrier")], None)])
  return copies

def nvme_submit(s:UOp) -> UOp|None:
  drives = {i: d for i, u in enumerate(s.without_after.body.src) if u.op is Ops.CALL and u.src[0].op is Ops.STORE and
            (d:=to_tuple(get_call_arg_uops(u)[1].device)[0]).startswith("NVME")}
  return copy_submit(s, drives, nvme_copies)
Compiled.pm_encode = PatternMatcher([(UPat.sink().call(UPat(Ops.PARAM)).after(allow_any_len=True, name="s"), nvme_submit)]) + Compiled.pm_encode
