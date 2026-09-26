import ctypes, struct
from dataclasses import dataclass
from tinygrad.helpers import ceildiv, getenv, wait_cond, DEBUG
from tinygrad.runtime.autogen import bnxt, pci
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.system import PCIDevice, System

BNXT_DEBUG = getenv("BNXT_DEBUG", 0)
BNXT_ACCESS, BNXT_INIT_MASK, BNXT_RTR_MASK, BNXT_RTS_MASK = 3, 0xd, 0x41515ad, 0xae005
BNXT_CHIMP_COMM, BNXT_CHIMP_COMM_TRIGGER = 0x0, 0x100
BNXT_BACKING_STORE = ((0, 64), (1, 0), (2, 128), (3, 0), (4, 2), (5, 0), (6, 0), (14, 1024), (15, 0))
WQE_SIZE, RING_ENTRIES, CQ_ENTRIES, MTU = 128, 1024, 1024, 4096
def db_value(xid, typ, index, epoch):
  return (xid & bnxt.DBC_DBC_XID_MASK | bnxt.DBC_DBC_PATH_ROCE | typ | bnxt.BNXT_QPLIB_DBR_VALID) << 32 | \
         index & bnxt.DBC_DBC_INDEX_MASK | epoch << bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT

# a wqe: type, flags, size in 16 byte units, then the send header (length at dword 1) or the receive header, then the sge (va, key, size)
def send_wqe(va:int, key:int, size:int) -> bytes: return struct.pack("<BBB5xI20xQII", 0, bnxt.SQ_SEND_FLAGS_SIGNAL_COMP, 3, size, va, key, size)
def recv_wqe(va:int, key:int, size:int) -> bytes: return struct.pack("<BBB29xQII", 0x80, 0, 3, va, key, size)
def msn_entry(wqe_idx:int, psn:int, size:int) -> tuple[int, int]: # the entry and the psn after this send
  psn &= 0xffffff
  nxt = (psn + max(1, ceildiv(size, MTU))) & 0xffffff
  # Thor2 static-mode retransmission indexes WQEs, not 16B slots.
  return (wqe_idx % RING_ENTRIES) << bnxt.SQ_MSN_SEARCH_START_IDX_SFT | nxt << bnxt.SQ_MSN_SEARCH_NEXT_PSN_SFT | psn, nxt
def cqe_ready(cqe:bytes, cons:int) -> bool: return (cqe[24] & bnxt.CQ_BASE_TOGGLE) != (cons // CQ_ENTRIES & 1)

def build_pbl(dev, paddrs, queue=False) -> tuple[int, int]:
  if len(paddrs) == 1: return 0, paddrs[0]
  values = [p | bnxt.PTU_PTE_VALID for p in paddrs]
  if queue: values[-1], values[-2] = values[-1] | bnxt.PTU_PTE_LAST, values[-2] | bnxt.PTU_PTE_NEXT_TO_LAST
  table, table_paddrs = dev.pci_dev.alloc_sysmem(ceildiv(len(values), 512) * 0x1000)
  table[:len(values) * 8] = struct.pack(f"<{len(values)}Q", *values)
  if len(table_paddrs) == 1: return 1, table_paddrs[0]
  assert len(table_paddrs) <= 512, f"a pbl has two levels: {len(values)} pages need bigger pages"
  top, top_paddrs = dev.pci_dev.alloc_sysmem(0x1000)
  top[:len(table_paddrs) * 8] = struct.pack(f"<{len(table_paddrs)}Q", *(p | bnxt.PTU_PTE_VALID for p in table_paddrs))
  return 2, top_paddrs[0]

@dataclass
class BNXTQueue:
  ring:MMIOInterface; paddrs:list[int]; stride:int; pbl_level:int; pbl_addr:int; size:int=0x1000; write_idx:int=0; read_idx:int=0 # noqa: E702
  def read(self, i:int) -> bytes: return bytes(self.ring[(off:=i % (self.size // self.stride) * self.stride):off + self.stride])
  def write(self, i:int, data:bytes, aux=False): # aux: the msn table after the ring
    off = self.size + i % (self.size // self.stride) * 8 if aux else i % (self.size // self.stride) * self.stride
    self.ring[off:off + len(data)] = data

def alloc_queue(dev, stride:int=16, aux=False, entries:int=0) -> BNXTQueue: # a page of entries by default
  entries = entries or 0x1000 // stride
  mem, paddrs = dev.pci_dev.alloc_sysmem((size:=entries * stride) + aux * entries * 8)
  return BNXTQueue(mem, paddrs, stride, *build_pbl(dev, paddrs, queue=True), size=size)

class BNXTDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.devfmt, self.seq = pci_dev, pci_dev.pcibus, 0
    pci_dev.reset()
    pci_dev.write_config(pci.PCI_COMMAND, pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.bar0, self.db = pci_dev.map_bar(0, fmt='I'), pci_dev.map_bar(2, fmt='Q')
    self.resp, self.resp_pa = pci_dev.alloc_sysmem(0x1000)

    ver = self.hwrm("ver_get")
    if DEBUG >= 2: print(f"bnxt {self.devfmt}: firmware {ver.hwrm_fw_maj_8b}.{ver.hwrm_fw_min_8b}.{ver.hwrm_fw_bld_8b}")
    self.hwrm("func_reset", timeout_ms=40000)
    caps = self.hwrm("func_qcaps", fid=0xffff)
    self.mac, self.port_id = int.from_bytes(bytes(caps.mac_address), 'big'), caps.port_id
    self.hwrm("func_drv_rgtr")
    self.db_off = self.hwrm("func_qcfg", fid=0xffff).legacy_l2_db_size_kb * 1024

    self.setup_backing_store()
    self._open_rcfw()
    self._open_l2()
    self.local_gid = bytes(10) + b'\xff\xff\x0a' + self.mac.to_bytes(6, 'big')[3:]
    self.gid_id = self.rcfw("add_gid", gid=struct.unpack(">4I", self.local_gid)[::-1], src_mac=struct.unpack(">3H", self.mac.to_bytes(6, 'big'))).xid

    if DEBUG >= 2: print(f"bnxt {self.devfmt}: booted mac={self.mac.to_bytes(6, 'big').hex(':')} gid={self.local_gid.hex()}")

  def hwrm(self, name, timeout_ms=10000, **fields):
    inp, out = getattr(bnxt, f"struct_hwrm_{name}_input"), getattr(bnxt, f"struct_hwrm_{name}_output")
    self.seq = (self.seq + 1) & 0xffff
    data = bytes(inp(req_type=getattr(bnxt, f"HWRM_{name.upper()}"), cmpl_ring=bnxt.BNXT_HWRM_NO_CMPL_RING,
                     seq_id=self.seq, target_id=bnxt.BNXT_HWRM_TARGET, resp_addr=self.resp_pa[0], **fields))
    self.resp[:] = bytes(len(self.resp))
    System.memory_barrier()
    for i, w in enumerate(memoryview(data.ljust(bnxt.HWRM_MAX_REQ_LEN, b'\0')).cast('I')): self.bar0[BNXT_CHIMP_COMM // 4 + i] = w
    self.bar0[BNXT_CHIMP_COMM_TRIGGER // 4] = 1
    def hdr(): return bnxt.struct_hwrm_resp_hdr.from_buffer_copy(bytes(self.resp[:8]))
    wait_cond(lambda: (n := hdr().resp_len) and hdr().seq_id == self.seq and self.resp[n - 1], timeout_ms=timeout_ms, msg=f"HWRM {name}")
    ret = out.from_buffer_copy(bytes(self.resp[:ctypes.sizeof(out)]))
    assert ret.error_code == 0, f"HWRM {name}: {ret.error_code}"
    return ret

  def setup_backing_store(self):
    counts: dict[int, int] = {}
    for typ, extra in BNXT_BACKING_STORE:
      caps = self.hwrm("func_backing_store_qcaps_v2", type=typ)
      size, splits = caps.entry_size, tuple(getattr(caps, f"split_entry_{j}") for j in range(caps.subtype_valid_cnt))
      counts[typ] = n = counts[0] if typ == 15 else max(caps.min_num_entries, sum(splits) + extra)
      # a zero bitmap means the type has a single instance 0
      for instance in [i for i in range(8) if caps.instance_bit_map >> i & 1] or [0]:
        mem, paddrs = self.pci_dev.alloc_sysmem(ceildiv(n * size, 0x1000) * 0x1000)
        if caps.ctx_init_value:
          init = bytearray(len(mem))
          init[caps.ctx_init_offset::size] = bytes([caps.ctx_init_value]) * len(range(caps.ctx_init_offset, len(mem), size))
          mem[:] = init
        lvl, base = build_pbl(self, paddrs)
        self.hwrm("func_backing_store_cfg_v2", type=typ, instance=instance, entry_size=size, num_entries=n, page_dir=base,
          page_size_pbl_level=lvl, subtype_valid_cnt=len(splits),
          flags=bnxt.FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_CFG_ALL_DONE if typ == 15 else 0,
          **{f"split_entry_{j}": v for j, v in enumerate(splits)})

  def _open_rcfw(self):
    self.rcfw_first, self.creq, self.cmdq = True, alloc_queue(self), alloc_queue(self)
    self.creq_id = self.hwrm("ring_alloc", ring_type=bnxt.RING_ALLOC_REQ_RING_TYPE_NQ, page_tbl_addr=self.creq.pbl_addr,
      page_size=12, page_tbl_depth=self.creq.pbl_level, length=256, int_mode=bnxt.RING_ALLOC_REQ_INT_MODE_MSIX).ring_id

    self.doorbell(self.creq_id, bnxt.DBC_DBC_TYPE_NQ_ARM, 0, 0)
    init = bnxt.struct_cmdq_init(cmdq_pbl=self.cmdq.pbl_addr, creq_ring_id=self.creq_id, cmdq_size_cmdq_lvl=256 << bnxt.CMDQ_INIT_CMDQ_SIZE_SFT)

    System.memory_barrier()
    for i, w in enumerate(memoryview(bytes(init)).cast('I')): self.bar0[bnxt.RCFW_COMM_BASE_OFFSET // 4 + i] = w

    _, p = self.pci_dev.alloc_sysmem(0x1000)
    self.rcfw("initialize_fw", stat_ctx_id=self.hwrm("stat_ctx_alloc", stats_dma_addr=p[0], stats_dma_length=176).stat_ctx_id,
      flags=bnxt.CMDQ_INITIALIZE_FW_FLAGS_HW_REQUESTER_RETX_SUPPORTED)

    # RoCE notification ring: never armed or serviced, but CQ and L2 ring allocation require one
    nq = alloc_queue(self)
    self.nq_id = self.hwrm("ring_alloc", ring_type=bnxt.RING_ALLOC_REQ_RING_TYPE_NQ, page_tbl_addr=nq.pbl_addr,
      page_size=12, page_tbl_depth=nq.pbl_level, length=16, logical_id=1, int_mode=bnxt.RING_ALLOC_REQ_INT_MODE_MSIX).ring_id

  def rcfw(self, name, timeout_ms=20000, **fields):
    req_t, resp_t = getattr(bnxt, f"struct_cmdq_{name}"), getattr(bnxt, f"struct_creq_{name}_resp")
    data = bytes(req_t(opcode=getattr(bnxt, f"CMDQ_BASE_OPCODE_{name.upper()}"),
                       cmd_size=(slots := ceildiv(ctypes.sizeof(req_t), 16)), **fields)).ljust(slots * 16, b'\0')
    for i in range(slots): self.cmdq.write(self.cmdq.write_idx + i, data[i * 16:(i + 1) * 16])

    self.cmdq.write_idx += slots
    prod = self.cmdq.write_idx & 0xffff
    if self.rcfw_first: prod, self.rcfw_first = prod | 1 << bnxt.FIRMWARE_FIRST_FLAG, False

    System.memory_barrier()

    self.bar0[(bnxt.RCFW_COMM_BASE_OFFSET + bnxt.RCFW_PF_VF_COMM_PROD_OFFSET) // 4] = prod
    self.bar0[(bnxt.RCFW_COMM_BASE_OFFSET + bnxt.RCFW_COMM_TRIG_OFFSET) // 4] = bnxt.RCFW_CMDQ_TRIG_VAL

    wait_cond(lambda: (self.creq.read(self.creq.read_idx)[8] & bnxt.CREQ_BASE_V) != (self.creq.read_idx // 256 & 1),
              timeout_ms=timeout_ms, msg=f"RCFW {name}")

    ret = resp_t.from_buffer_copy(self.creq.read(self.creq.read_idx))
    self.creq.read_idx += 1

    # NQ_ARM also publishes the CREQ consumer index, which is what frees ring space for the next command
    self.doorbell(self.creq_id, bnxt.DBC_DBC_TYPE_NQ_ARM, self.creq.read_idx & 255, (self.creq.read_idx // 256) & 1)
    assert ret.status == 0, f"RCFW {name}: {ret.status}"

    if BNXT_DEBUG >= 1: print(f"bnxt {self.devfmt}: rcfw {name} xid={getattr(ret, 'xid', 0):#x}")
    return ret

  def fini(self): self.hwrm("func_drv_unrgtr")

  def doorbell(self, xid, typ, index, epoch):
    System.memory_barrier()
    self.db[self.db_off // 8] = db_value(xid, typ, index, epoch)

  # L2 receive path, required for RoCE ingress even though no ethernet receive buffers are posted
  def _open_l2(self):
    cq = alloc_queue(self)
    ci = self.hwrm("ring_alloc", enables=bnxt.RING_ALLOC_REQ_ENABLES_NQ_RING_ID_VALID, ring_type=bnxt.RING_ALLOC_REQ_RING_TYPE_L2_CMPL,
      page_tbl_addr=cq.pbl_addr, page_size=12, page_tbl_depth=cq.pbl_level, length=16, nq_ring_id=self.nq_id).ring_id
    rx = alloc_queue(self)
    ri = self.hwrm("ring_alloc", enables=bnxt.RING_ALLOC_REQ_ENABLES_NQ_RING_ID_VALID |
      bnxt.RING_ALLOC_REQ_ENABLES_RX_BUF_SIZE_VALID, ring_type=bnxt.RING_ALLOC_REQ_RING_TYPE_RX, page_tbl_addr=rx.pbl_addr,
      page_size=12, page_tbl_depth=rx.pbl_level, length=16, rx_buf_size=640, nq_ring_id=self.nq_id).ring_id
    vi = self.hwrm("vnic_alloc").vnic_id
    self.hwrm("vnic_cfg", enables=bnxt.VNIC_CFG_REQ_ENABLES_MRU | bnxt.VNIC_CFG_REQ_ENABLES_DEFAULT_RX_RING_ID |
      bnxt.VNIC_CFG_REQ_ENABLES_DEFAULT_CMPL_RING_ID, vnic_id=vi, mru=9018, default_rx_ring_id=ri, default_cmpl_ring_id=ci)
    self.hwrm("cfa_l2_filter_alloc", flags=bnxt.CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_RX,
      enables=bnxt.CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR | bnxt.CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR_MASK |
      bnxt.CFA_L2_FILTER_ALLOC_REQ_ENABLES_DST_ID, l2_addr=tuple(self.mac.to_bytes(6, 'big')), l2_addr_mask=(0xff,) * 6, dst_id=vi)

  # a memory region over pages: the key addresses [va, va + size) as those pages
  def register_mem(self, paddrs:list[int], size:int, log_page_size:int=12, va:int|None=None) -> int:
    level, base = build_pbl(self, paddrs[:ceildiv(size, 1 << log_page_size)])
    return self.rcfw("register_mr", flags=bnxt.CMDQ_REGISTER_MR_FLAGS_ALLOC_MR,
      log2_pg_size_lvl=level << bnxt.CMDQ_REGISTER_MR_LVL_SFT | log_page_size << bnxt.CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT,
      access=bnxt.CMDQ_REGISTER_MR_ACCESS_LOCAL_WRITE | bnxt.CMDQ_REGISTER_MR_ACCESS_REMOTE_WRITE,
      log2_pbl_pg_size=12, pbl=base, va=paddrs[0] if va is None else va, mr_size=size).xid
  def unregister_mem(self, key:int): self.rcfw("deregister_mr", lkey=key)

class BNXTQP:
  def __init__(self, dev:BNXTDev):
    self.dev, self.sq_psn = dev, 0
    self.scq, self.rcq = (alloc_queue(dev, ctypes.sizeof(bnxt.struct_cq_base), entries=CQ_ENTRIES) for _ in range(2))
    self.scq_id, self.rcq_id = (dev.rcfw("create_cq", cq_size=CQ_ENTRIES, pbl=q.pbl_addr, pg_size_lvl=q.pbl_level, cq_fco_cnq_id=dev.nq_id).xid
                                for q in (self.scq, self.rcq))
    self.sq, self.rq = alloc_queue(dev, WQE_SIZE, aux=True, entries=RING_ENTRIES), alloc_queue(dev, WQE_SIZE, entries=RING_ENTRIES)
    self.qpn = dev.rcfw("create_qp", type=bnxt.CMDQ_CREATE_QP_TYPE_RC, scq_cid=self.scq_id, rcq_cid=self.rcq_id,
      sq_size=RING_ENTRIES, sq_fwo_sq_sge=6, sq_pbl=self.sq.pbl_addr, sq_pg_size_sq_lvl=self.sq.pbl_level,
      rq_size=RING_ENTRIES, rq_fwo_rq_sge=6, rq_pbl=self.rq.pbl_addr, rq_pg_size_rq_lvl=self.rq.pbl_level).xid
    self.modify_qp(1, BNXT_INIT_MASK, access=BNXT_ACCESS, pkey=0xffff)

  def modify_qp(self, state, mask, network_type=0, **fields):
    self.dev.rcfw("modify_qp", qp_cid=self.qpn, modify_mask=mask, network_type_en_sqd_async_notify_new_state=state | network_type, **fields)

  def connect(self, qpn:int, gid:bytes, mac:int):
    network_type = bnxt.CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV4
    dgid, dmac = struct.unpack("<4I", gid), struct.unpack("<3H", mac.to_bytes(6, 'big'))

    self.modify_qp(2, BNXT_RTR_MASK, network_type=network_type, qp_type=bnxt.CMDQ_MODIFY_QP_QP_TYPE_RC, access=BNXT_ACCESS,
      pkey=0xffff, dgid=dgid, sgid_index=self.dev.gid_id, hop_limit=64, dest_mac=dmac, min_rnr_timer=1,
      path_mtu_pingpong_push_enable=bnxt.CMDQ_MODIFY_QP_PATH_MTU_MTU_4096, max_dest_rd_atomic=4, dest_qp_id=qpn)
    # a send to a not yet posted receive is retried forever
    self.modify_qp(3, BNXT_RTS_MASK, network_type=network_type, qp_type=bnxt.CMDQ_MODIFY_QP_QP_TYPE_RC, access=BNXT_ACCESS,
      max_rd_atomic=1, rnr_retry=7, retry_cnt=7, timeout=14)

    if BNXT_DEBUG >= 1: print(f"bnxt: QP {self.qpn:#x} connected (remote={qpn:#x})")

  def poll(self, cq:BNXTQueue, cq_id, timeout_ms=20000) -> bytes:
    wait_cond(lambda: cqe_ready(cq.read(cq.read_idx), cq.read_idx), timeout_ms=timeout_ms, msg="BNXT CQ")
    raw = cq.read(cq.read_idx)
    cq.read_idx += 1
    self.dev.doorbell(cq_id, bnxt.DBC_DBC_TYPE_CQ, cq.read_idx % CQ_ENTRIES, (cq.read_idx // CQ_ENTRIES) & 1)
    assert raw[25] == 0, f"BNXT CQE status {raw[25]}"
    return raw

  def post_send(self, va:int, key:int, size:int):
    self.sq.write(self.sq.write_idx, send_wqe(va, key, size))
    entry, self.sq_psn = msn_entry(self.sq.write_idx, self.sq_psn, size)
    self.sq.write(self.sq.write_idx, struct.pack("<Q", entry), aux=True)
    self.sq.write_idx += 1
    self.dev.doorbell(self.qpn, bnxt.DBC_DBC_TYPE_SQ, self.sq.write_idx % RING_ENTRIES, (self.sq.write_idx // RING_ENTRIES) & 1)

  def post_recv(self, va:int, key:int, size:int):
    self.rq.write(self.rq.write_idx, recv_wqe(va, key, size))
    self.rq.write_idx += 1
    self.dev.doorbell(self.qpn, bnxt.DBC_DBC_TYPE_RQ, self.rq.write_idx % RING_ENTRIES, (self.rq.write_idx // RING_ENTRIES) & 1)
