import struct, unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinygrad.runtime.autogen import bnxt
from extra.bnxt_driver.bnxtdev import BNXT_BACKING_STORE, BNXTDev, BNXTQP, _queue, _qwrite, ipv4_to_gid

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
    return SimpleNamespace(xid={"create_cq":77, "create_qp":88, "register_mr":0x5678}.get(name, 0))
  def doorbell(self, *args, **kwargs): self.doorbells.append((args, kwargs))

class FakeQPDev:
  def __init__(self): self.pci_dev, self.fw, self.gid_id, self.nq_id = FakePCI(), FakeRCFW(), 9, 41
  def rcfw(self, *args, **kwargs): return self.fw.exec(*args, **kwargs)
  def doorbell(self, *args, **kwargs): self.fw.doorbell(*args, **kwargs)

class TestMemory(unittest.TestCase):
  def test_cmdq_and_sq_aux(self):
    dev = FakeDev()
    cmdq, sq = _queue(dev), _queue(dev, aux=True)
    self.assertEqual((cmdq["level"], cmdq["base"]), (0, 0x100000))
    _qwrite(sq, 3, b"ABCDEFGH", aux=True)
    self.assertEqual(bytes(sq["mem"][0x1018:0x1020]), b"ABCDEFGH")

  def test_f320_backing_layout_and_final_marker(self):
    self.assertEqual(len(BNXT_BACKING_STORE), 9)
    dev = FakeDev()
    small = ((0, 6), (15, 0))
    with patch("extra.bnxt_driver.bnxtdev.BNXT_BACKING_STORE", small): BNXTDev.setup_backing_store(dev)
    cfg = [fields for name, fields in dev.calls if name == "func_backing_store_cfg_v2"]
    self.assertEqual([(x["type"], x["instance"]) for x in cfg], [(0, 0), (0, 2), (15, 0)])
    self.assertTrue(all(not x["flags"] for x in cfg[:-1]))
    self.assertEqual(cfg[-1]["flags"], bnxt.FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_CFG_ALL_DONE)
    self.assertEqual((dev.pci_dev.allocations[0][4], dev.pci_dev.allocations[0][20]), (0x5a, 0x5a))

class TestRCFW(unittest.TestCase):
  def setUp(self):
    patch("extra.bnxt_driver.bnxtdev.System.memory_barrier").start()
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
    dev.bar0, dev.cmdq, dev.creq = [0]*1024, _queue(FakeDev()), _queue(FakeDev())
    dev.rcfw_first, dev.creq_id = True, 23
    dev.doorbell = lambda *args: None
    _qwrite(dev.creq, 0, bytes(bnxt.struct_creq_query_version_resp(type=bnxt.CREQ_BASE_TYPE_QP_EVENT, cookie=0, v=1)))
    ret = dev.rcfw("query_version")
    req = bnxt.struct_cmdq_query_version.from_buffer_copy(bytes(dev.cmdq["mem"][:16]))
    prod = dev.bar0[(bnxt.RCFW_COMM_BASE_OFFSET+bnxt.RCFW_PF_VF_COMM_PROD_OFFSET)//4]
    self.assertEqual((req.cookie, ret.cookie, prod), (0, 0, 1 | 1<<bnxt.FIRMWARE_FIRST_FLAG))

class TestFastPath(unittest.TestCase):
  def test_unified_mr(self):
    dev = BNXTDev.__new__(BNXTDev)
    fw = FakeRCFW()
    dev.pci_dev, dev.rcfw = FakePCI(), fw.exec
    self.assertEqual(dev.register_mem([0x800000, 0x900000], 0x2000), 0x5678)
    mr = fw.calls[-1][1]
    self.assertEqual((mr["flags"], mr["va"], mr["mr_size"], mr["log2_pg_size_lvl"]),
      (bnxt.CMDQ_REGISTER_MR_FLAGS_ALLOC_MR, 0x800000, 0x2000,
       1<<bnxt.CMDQ_REGISTER_MR_LVL_SFT | 12<<bnxt.CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT))

  def test_qp_creation_and_connect_use_f320_layout(self):
    dev = FakeQPDev()
    qp = BNXTQP(dev)
    create = next(fields for name, fields in dev.fw.calls if name == "create_qp")
    self.assertEqual((create["sq_size"], "rq_size" in create, qp.qpn), (16, False, 88))
    qp.connect(0x123, ipv4_to_gid("10.0.0.2"), 0x001122334455)
    rtr, rts = dev.fw.calls[-2][1], dev.fw.calls[-1][1]
    self.assertEqual((bytes(rtr["dgid"]), bytes(rtr["dest_mac"])),
      (ipv4_to_gid("10.0.0.2"), bytes.fromhex("001122334455")))
    self.assertEqual((rtr["modify_mask"], rts["modify_mask"]), (0x41515ad, 0xae005))

  def test_rdma_write_builds_three_slots_and_host_msn(self):
    qp = BNXTQP.__new__(BNXTQP)
    qp.dev, qp.qpn = FakeQPDev(), 88
    qp.sq, qp.sq_psn, qp.msn = _queue(FakeDev(), aux=True), 5, 0
    qp._poll = lambda timeout: bytes(bnxt.struct_cq_req())
    qp.rdma_write(0x1122334455667788, 0x99aa, 0x12345000, 0x55aa, 100)
    hdr = bnxt.struct_sq_rdma_hdr.from_buffer_copy(bytes(qp.sq["mem"][:32]))
    sge = bnxt.struct_sq_sge.from_buffer_copy(bytes(qp.sq["mem"][32:48]))
    self.assertEqual((hdr.remote_va, hdr.remote_key, hdr.length, sge.va_or_pa, sge.l_key, sge.size),
      (0x1122334455667788, 0x99aa, 100, 0x12345000, 0x55aa, 100))
    self.assertEqual(struct.unpack_from("<Q", qp.sq["mem"], 0x1000)[0], 6<<24 | 5)
    self.assertEqual(qp.dev.fw.doorbells, [((88, bnxt.DBC_DBC_TYPE_SQ, 3, 0), {})])

if __name__ == "__main__": unittest.main()
