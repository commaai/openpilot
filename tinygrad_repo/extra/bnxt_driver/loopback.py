#!/usr/bin/env python3
"""CPU-driven SEND/RECV through PHY loopback. Run with PYTHONPATH=."""
import atexit, time
from tinygrad.helpers import getenv
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import BNXTDev, BNXTQP
from tinygrad.runtime.support.system import PCIDevice

if __name__ == "__main__":
  dev = BNXTDev(PCIDevice("bnxt", getenv("BNXT_PCI", "0000:41:00.0")))
  atexit.register(dev.fini)
  tx, rx = BNXTQP(dev), BNXTQP(dev)
  tx.connect(rx.qpn, dev.local_gid, dev.mac)
  rx.connect(tx.qpn, dev.local_gid, dev.mac)
  src, src_pages = dev.pci_dev.alloc_sysmem(size := getenv("SIZE", 0x2000))
  dst, dst_pages = dev.pci_dev.alloc_sysmem(size)
  skey, dkey = dev.register_mem(src_pages, size), dev.register_mem(dst_pages, size)
  dev.hwrm("port_phy_cfg", port_id=dev.port_id, enables=bnxt.PORT_PHY_CFG_REQ_ENABLES_LPBK, lpbk=bnxt.PORT_PHY_CFG_REQ_LPBK_LOCAL)
  try:
    time.sleep(1)
    for i in range(getenv("ITERS", 257)):
      src[:], dst[:] = (message := bytes([i % 255 + 1]) * size), bytes(size)
      if i % 2:  # Force RNR retries at nonzero MSN indices, including after ring wrap.
        tx.post_send(src_pages[0], skey, size)
        time.sleep(0.01)
        rx.post_recv(dst_pages[0], dkey, size)
      else:
        rx.post_recv(dst_pages[0], dkey, size)
        tx.post_send(src_pages[0], skey, size)
      tx.poll(tx.scq, tx.scq_id)
      rx.poll(rx.rcq, rx.rcq_id)
      assert bytes(dst[:]) == message, f"loopback mismatch at iteration {i}"
    print(f"BNXT SEND/RECV loopback passed: {i + 1} x {size} bytes")
  finally:
    dev.hwrm("port_phy_cfg", port_id=dev.port_id, enables=bnxt.PORT_PHY_CFG_REQ_ENABLES_LPBK, lpbk=bnxt.PORT_PHY_CFG_REQ_LPBK_NONE)
    dev.unregister_mem(skey)
    dev.unregister_mem(dkey)
