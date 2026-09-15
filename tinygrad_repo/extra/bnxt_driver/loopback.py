#!/usr/bin/env python3
"""Local BNXT RoCEv2 RDMA WRITE loopback using the firmware's PHY loopback mode.

The kernel bnxt_en/bnxt_re modules must be unloaded first.

  sudo PYTHONPATH=. BNXT_PCI=0000:41:00.0 BNXT_IP=10.0.200.5 python3 extra/bnxt_driver/loopback.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from extra.bnxt_driver.bnxtdev import BNXTDev, BNXTQP
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.system import PCIDevice

BUF_SIZE = 0x1000
BNXT_PCI = os.getenv("BNXT_PCI", "0000:41:00.0")
BNXT_IP = os.getenv("BNXT_IP", "10.0.200.5")

if __name__ == "__main__":
  print(f"[init] BNXT at {BNXT_PCI}")
  dev = BNXTDev(PCIDevice("bnxt", BNXT_PCI), ip=BNXT_IP)
  tx_qp, rx_qp = BNXTQP(dev), BNXTQP(dev)
  print(f"[init] loopback-connect TX QP 0x{tx_qp.qpn:x} <-> RX QP 0x{rx_qp.qpn:x}")
  tx_qp.connect(rx_qp.qpn, dev.local_gid, dev.mac)
  rx_qp.connect(tx_qp.qpn, dev.local_gid, dev.mac)

  src, src_paddrs = dev.pci_dev.alloc_sysmem(BUF_SIZE)
  dst, dst_paddrs = dev.pci_dev.alloc_sysmem(BUF_SIZE)
  message = b"Hello from BNXT RoCE PHY loopback!"
  src[:BUF_SIZE], dst[:BUF_SIZE] = bytes(BUF_SIZE), bytes(BUF_SIZE)
  src[:len(message)] = message
  lkey = dev.register_mem(src_paddrs, BUF_SIZE)
  rkey = dev.register_mem(dst_paddrs, BUF_SIZE)

  print("[loopback] enabling local PHY loopback")
  dev.hwrm("port_phy_cfg", port_id=dev.port_id, enables=bnxt.PORT_PHY_CFG_REQ_ENABLES_LPBK, lpbk=bnxt.PORT_PHY_CFG_REQ_LPBK_LOCAL)
  time.sleep(1)
  tx_qp.rdma_write(dst_paddrs[0], rkey, src_paddrs[0], lkey, len(message))
  got = bytes(dst[:len(message)])
  print(f"[result] {got!r}")
  assert got == message
  print("BNXT RoCE PHY loopback RDMA WRITE passed")
  dev.hwrm("port_phy_cfg", port_id=dev.port_id, enables=bnxt.PORT_PHY_CFG_REQ_ENABLES_LPBK, lpbk=bnxt.PORT_PHY_CFG_REQ_LPBK_NONE)
