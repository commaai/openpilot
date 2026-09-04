import ctypes, unittest
from unittest.mock import patch

from tinygrad.runtime.autogen import bnxt
from extra.bnxt_driver.bnxtdev import BNXT_CHIMP_COMM, BNXT_CHIMP_COMM_TRIGGER, BNXTDev

class Mailbox:
  def __init__(self, trigger): self.words, self.trigger = {}, trigger
  def __setitem__(self, idx, val):
    self.words[idx] = val
    if idx == BNXT_CHIMP_COMM_TRIGGER//4: self.trigger()
  def request(self):
    base = BNXT_CHIMP_COMM//4
    return b"".join(self.words.get(base+i, 0).to_bytes(4, "little") for i in range(bnxt.HWRM_MAX_REQ_LEN//4))

def fake_dev():
  dev = BNXTDev.__new__(BNXTDev)
  dev.resp, dev.resp_pa, dev.seq = bytearray(0x1000), [0x6789a000], 0
  return dev

def reply(dev, out_type):
  req = bnxt.struct_hwrm_cmd_hdr.from_buffer_copy(dev.bar0.request())
  out = out_type(req_type=req.req_type, seq_id=req.seq_id, resp_len=ctypes.sizeof(out_type), valid=1)
  dev.resp[:ctypes.sizeof(out_type)] = bytes(out)

class TestHWRM(unittest.TestCase):
  def setUp(self):
    self.barrier = patch("extra.bnxt_driver.bnxtdev.System.memory_barrier").start()
    self.addCleanup(patch.stopall)

  def test_request(self):
    dev = fake_dev()
    dev.bar0 = Mailbox(lambda: reply(dev, bnxt.struct_hwrm_func_qcaps_output))
    dev.hwrm("func_qcaps", fid=0xffff)
    req = bnxt.struct_hwrm_func_qcaps_input.from_buffer_copy(dev.bar0.request())
    self.assertEqual((req.req_type, req.seq_id, req.resp_addr, req.fid),
      (bnxt.HWRM_FUNC_QCAPS, 1, dev.resp_pa[0], 0xffff))
    self.barrier.assert_called_once_with()

if __name__ == "__main__": unittest.main()
