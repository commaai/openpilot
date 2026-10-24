import struct, unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import BNXT_BACKING_STORE, BNXTDev, BNXTQP, alloc_queue, cqe_ready, msn_entry
from tinygrad.runtime.support.rdma.bnxtdev import RING_ENTRIES, CQ_ENTRIES, WQE_SIZE
from tinygrad.runtime.support.system import ipv4_to_gid

class FakePCI:
  def __init__(self): self.next_addr, self.allocations = 0x100000, []
  def alloc_sysmem(self, size, contiguous=False):
    pages = [self.next_addr+i*0x1000 for i in range((size+0xfff)//0x1000)]
    self.next_addr += len(pages)*0x1000
    self.allocations.append(mem := bytearray(size))
    return mem, pages

class FakeDev:
  def __init__(self): self.pci_dev, self.calls = FakePCI(), []
  def hwrm(self, name, **fields):
    self.calls.append((name, fields))
    typ = fields.get("type", 0)
    return SimpleNamespace(ctx_init_value=0x5a, ctx_init_offset=4, entry_size=16 if typ == 0 else 4,
                           subtype_valid_cnt=typ == 0, split_entry_0=2, instance_bit_map=5 if typ == 0 else 1, min_num_entries=0)

class FakeRCFW:
  def __init__(self): self.calls, self.doorbells = [], []
  def exec(self, name, **fields):
    self.calls.append((name, fields))
    return SimpleNamespace(xid={"create_cq":77 + sum(n == "create_cq" for n, _ in self.calls), "create_qp":88, "register_mr":0x5678}.get(name, 0))
  def doorbell(self, *args, **kwargs): self.doorbells.append((args, kwargs))

class FakeQPDev:
  def __init__(self): self.pci_dev, self.fw, self.gid_id, self.nq_id = FakePCI(), FakeRCFW(), 9, 41
  def rcfw(self, *args, **kwargs): return self.fw.exec(*args, **kwargs)
  def doorbell(self, *args, **kwargs): self.fw.doorbell(*args, **kwargs)

class TestMemory(unittest.TestCase):
  def test_cmdq_and_sq_aux(self):
    dev = FakeDev()
    cmdq, sq = alloc_queue(dev), alloc_queue(dev, aux=True)
    self.assertEqual((cmdq.pbl_level, cmdq.pbl_addr), (0, 0x100000))
    self.assertEqual(len(sq.ring), sq.size + (sq.size // sq.stride) * 8)
    sq.write(3, b"ABCDEFGH", aux=True)
    self.assertEqual(bytes(sq.ring[0x1018:0x1020]), b"ABCDEFGH")

  def test_f320_backing_layout_and_final_marker(self):
    self.assertEqual(len(BNXT_BACKING_STORE), 9)
    dev = FakeDev()
    small = ((0, 6), (15, 0))
    with patch("tinygrad.runtime.support.rdma.bnxtdev.BNXT_BACKING_STORE", small): BNXTDev.setup_backing_store(dev)
    cfg = [fields for name, fields in dev.calls if name == "func_backing_store_cfg_v2"]
    self.assertEqual([(x["type"], x["instance"]) for x in cfg], [(0, 0), (0, 2), (15, 0)])
    self.assertTrue(all(not x["flags"] for x in cfg[:-1]))
    self.assertEqual(cfg[-1]["flags"], bnxt.FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_CFG_ALL_DONE)
    self.assertEqual(dev.pci_dev.allocations[0][:32], (bytes(4) + b"\x5a" + bytes(11)) * 2)

class TestRCFW(unittest.TestCase):
  def setUp(self):
    patch("tinygrad.runtime.support.rdma.bnxtdev.System.memory_barrier").start()
    self.addCleanup(patch.stopall)

  def test_doorbell_encodes_xid_type_and_index(self):
    dev = BNXTDev.__new__(BNXTDev)
    dev.db, dev.db_off = [0]*1024, 0x1000
    dev.doorbell(0x123456, bnxt.DBC_DBC_TYPE_CQ_ARMALL, 0x456, epoch=1)
    key = dev.db[0x1000//8]
    self.assertEqual(key >> 32,
      0x123456 & bnxt.DBC_DBC_XID_MASK | bnxt.DBC_DBC_PATH_ROCE | bnxt.DBC_DBC_TYPE_CQ_ARMALL | bnxt.BNXT_QPLIB_DBR_VALID)
    self.assertEqual(key & 0xffffffff, 0x456 | 1<<bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT)

  def test_command_uses_first_flag(self):
    dev = BNXTDev.__new__(BNXTDev)
    dev.bar0, dev.cmdq, dev.creq = [0]*1024, alloc_queue(FakeDev()), alloc_queue(FakeDev())
    dev.rcfw_first, dev.creq_id = True, 23
    dev.doorbell = lambda *args: None
    dev.creq.write(0, bytes(bnxt.struct_creq_query_version_resp(type=bnxt.CREQ_BASE_TYPE_QP_EVENT, cookie=0, v=1)))
    ret = dev.rcfw("query_version")
    req = bnxt.struct_cmdq_query_version.from_buffer_copy(bytes(dev.cmdq.ring[:16]))
    prod = dev.bar0[(bnxt.RCFW_COMM_BASE_OFFSET+bnxt.RCFW_PF_VF_COMM_PROD_OFFSET)//4]
    self.assertEqual((req.cookie, ret.cookie, prod), (0, 0, 1 | 1<<bnxt.FIRMWARE_FIRST_FLAG))

  def test_firmware_queue_wrap(self):
    dev = BNXTDev.__new__(BNXTDev)
    dev.bar0, dev.cmdq, dev.creq = [0]*1024, alloc_queue(FakeDev()), alloc_queue(FakeDev())
    dev.rcfw_first, dev.creq_id, dev.devfmt = True, 23, "test"
    bells = []
    dev.doorbell = lambda *args: bells.append(args)
    for i in range(513):
      dev.creq.write(i, bytes(bnxt.struct_creq_query_version_resp(v=(i // 256 & 1) ^ 1)))
      dev.rcfw("query_version", timeout_ms=100)
      self.assertEqual(bells[-1], (23, bnxt.DBC_DBC_TYPE_NQ_ARM, (i + 1) % 256, (i + 1) // 256 & 1))
    self.assertEqual((dev.creq.read_idx, dev.cmdq.write_idx), (513, 513))

class TestFastPath(unittest.TestCase):
  def test_firmware_layouts(self):
    self.assertEqual(len(bytes(bnxt.struct_cmdq_deregister_mr(lkey=0x1234))), 24)
    self.assertEqual(bytes(bnxt.struct_cmdq_deregister_mr(lkey=0x1234))[16:20], b"\x34\x12\x00\x00")
    self.assertEqual(len(bytes(bnxt.struct_creq_deregister_mr_resp())), 16)

  def test_unified_mr(self):
    dev = BNXTDev.__new__(BNXTDev)
    fw = FakeRCFW()
    dev.pci_dev, dev.rcfw = FakePCI(), fw.exec
    self.assertEqual(dev.register_mem([0x800000, 0x900000], 0x2000), 0x5678)
    mr = fw.calls[-1][1]
    self.assertEqual((mr["flags"], mr["va"], mr["mr_size"], mr["log2_pg_size_lvl"]),
      (bnxt.CMDQ_REGISTER_MR_FLAGS_ALLOC_MR, 0x800000, 0x2000,
       1<<bnxt.CMDQ_REGISTER_MR_LVL_SFT | 12<<bnxt.CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT))
    key = dev.register_mem([0x800000], 0x200000, 21, va=0x100000000)
    mr = fw.calls[-1][1]
    self.assertEqual((mr["va"], mr["pbl"], mr["log2_pg_size_lvl"]), (0x100000000, 0x800000, 21<<bnxt.CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT))
    dev.unregister_mem(key)
    self.assertEqual(fw.calls[-1], ("deregister_mr", {"lkey":key}))

  def test_qp_creation_and_connect_use_f320_layout(self):
    dev = FakeQPDev()
    qp = BNXTQP(dev)
    create = next(fields for name, fields in dev.fw.calls if name == "create_qp")
    self.assertEqual((create["sq_size"], create["rq_size"], create["sq_fwo_sq_sge"], create["rq_fwo_rq_sge"]), (1024, 1024, 6, 6))
    self.assertNotEqual(create["scq_cid"], create["rcq_cid"])
    self.assertEqual((qp.sq.stride, qp.rq.stride, qp.scq.stride, qp.rcq.stride), (128, 128, 32, 32))
    self.assertEqual((len(qp.sq.ring), len(qp.rq.ring), len(qp.scq.ring), len(qp.rcq.ring)), (139264, 131072, 32768, 32768))
    self.assertEqual([fields["cq_size"] for name, fields in dev.fw.calls if name == "create_cq"], [1024, 1024])
    qp.connect(0x123, ipv4_to_gid("10.0.0.2"), 0x001122334455)
    rtr, rts = dev.fw.calls[-2][1], dev.fw.calls[-1][1]
    cmd = bnxt.struct_cmdq_modify_qp(**rtr)
    self.assertEqual((bytes(cmd.dgid), bytes(cmd.dest_mac)),
      (ipv4_to_gid("10.0.0.2"), bytes.fromhex("001122334455")))
    self.assertTrue(rtr["modify_mask"] & bnxt.CMDQ_MODIFY_QP_MODIFY_MASK_MIN_RNR_TIMER)
    for mask in (bnxt.CMDQ_MODIFY_QP_MODIFY_MASK_TIMEOUT, bnxt.CMDQ_MODIFY_QP_MODIFY_MASK_RETRY_CNT,
                 bnxt.CMDQ_MODIFY_QP_MODIFY_MASK_RNR_RETRY): self.assertTrue(rts["modify_mask"] & mask)
    self.assertEqual((rtr["min_rnr_timer"], rts["rnr_retry"], rts["retry_cnt"], rts["timeout"]), (1, 7, 7, 14))

  def test_send_recv_ring_wrap(self):
    qp = BNXTQP(FakeQPDev())
    qp.sq_psn = 0xfffffe
    psn = qp.sq_psn
    for i in range(3 * RING_ENTRIES + 1):
      size = (i % 3) * 4096 + 1
      qp.post_send(0x12345000 + i, 0x55aa, size)
      qp.post_recv(0x22345000 + i, 0x66aa, size)
      off = i % RING_ENTRIES * WQE_SIZE
      self.assertEqual(bytes(qp.sq.ring[off:off + 4]), b"\x00\x01\x03\x00")
      self.assertEqual(struct.unpack_from("<I", qp.sq.ring, off + 8)[0], size)
      self.assertEqual(bytes(qp.rq.ring[off:off + 4]), b"\x80\x00\x03\x00")
      self.assertEqual(struct.unpack_from("<QII", qp.sq.ring, off + 32), (0x12345000 + i, 0x55aa, size))
      self.assertEqual(struct.unpack_from("<QII", qp.rq.ring, off + 32), (0x22345000 + i, 0x66aa, size))
      nxt = (psn + i % 3 + 1) & 0xffffff
      self.assertEqual(struct.unpack_from("<Q", qp.sq.ring, RING_ENTRIES * WQE_SIZE + i % RING_ENTRIES * 8)[0],
                       (i % RING_ENTRIES) << 48 | nxt << 24 | psn)
      self.assertEqual(qp.sq_psn, nxt)
      psn = nxt
      for typ, doorbell in zip((bnxt.DBC_DBC_TYPE_SQ, bnxt.DBC_DBC_TYPE_RQ), qp.dev.fw.doorbells[-2:]):
        self.assertEqual(doorbell, ((qp.qpn, typ, (i + 1) % RING_ENTRIES, (i + 1) // RING_ENTRIES & 1), {}))

  def test_zero_length_send_consumes_psn(self):
    self.assertEqual(msn_entry(RING_ENTRIES, 0x1ffffff, 0), (0xffffff, 0))

  def test_cq_wrap_and_status(self):
    qp = BNXTQP(FakeQPDev())
    for i in range(3 * CQ_ENTRIES + 1):
      cqe = bytearray(32)
      cqe[24] = (i // CQ_ENTRIES & 1) ^ 1
      self.assertTrue(cqe_ready(cqe, i))
      self.assertFalse(cqe_ready(cqe, i + CQ_ENTRIES))
      qp.scq.write(i, cqe)
      self.assertEqual(qp.poll(qp.scq, qp.scq_id), bytes(cqe))
      self.assertEqual(qp.dev.fw.doorbells[-1], ((qp.scq_id, bnxt.DBC_DBC_TYPE_CQ, (i + 1) % CQ_ENTRIES, (i + 1) // CQ_ENTRIES & 1), {}))
    cqe[25] = 1
    qp.scq.write(3 * CQ_ENTRIES + 1, cqe)
    with self.assertRaisesRegex(AssertionError, "BNXT CQE status 1"): qp.poll(qp.scq, qp.scq_id)

if __name__ == "__main__": unittest.main()
