#!/usr/bin/env python3
"""Send and validate one RDMA WRITE between two Broadcom BNXT hosts.

This follows ``extra/mlx_driver/connect.py``: sync the driver, start the remote
endpoint over SSH, exchange QP/GID/MAC/MR metadata, move both RC QPs to RTS,
write bytes into the remote MR, and verify the bytes on the remote host.

Both PCI functions must be unbound from bnxt_en/bnxt_re first.
"""
import json
import os
import subprocess
import sys
from typing import Any, IO

TINYGRAD = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, TINYGRAD)

from extra.bnxt_driver.bnxtdev import BNXTDev, BNXTQP
from tinygrad.runtime.support.system import PCIDevice

REMOTE_HOST = os.getenv("REMOTE_HOST", "192.168.52.213")
REMOTE_USER = os.getenv("REMOTE_USER", "nimlgen")
LOCAL_PCI = os.getenv("BNXT_PCI", "0000:41:00.0")
REMOTE_PCI = os.getenv("REMOTE_PCI", "0000:41:00.0")
LOCAL_IP = os.getenv("LOCAL_IP", "10.0.200.5")
REMOTE_IP = os.getenv("REMOTE_IP", "10.0.200.6")
MESSAGE = os.getenv("RDMA_MESSAGE", "Test message, rdma works!").encode()
REMOTE = f"{REMOTE_USER}@{REMOTE_HOST}"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", REMOTE]
SYNC_FILES = ("tinygrad/runtime/autogen/bnxt.py", "tinygrad/runtime/support/system.py",
              "extra/bnxt_driver/bnxtdev.py", "extra/bnxt_driver/connect.py")

def read_json(stream:IO[str], what:str) -> dict[str, Any]:
  for line in iter(stream.readline, ""):
    print(f"  [remote] {line}", end="")
    try: value = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(value, dict): return value
  raise RuntimeError(f"remote exited before publishing {what}")

def wait_line(stream:IO[str], text:str) -> str:
  for line in iter(stream.readline, ""):
    print(f"  [remote] {line}", end="")
    if text in line: return line
  raise RuntimeError(f"remote exited before reporting {text!r}")

def send_line(stream:IO[str], value:str|dict[str, Any]):
  stream.write((json.dumps(value) if isinstance(value, dict) else value) + "\n")
  stream.flush()

def qp_info(dev:BNXTDev, qp:BNXTQP) -> dict[str, Any]:
  return {"qpn":qp.qpn, "mac":dev.mac.to_bytes(6, "big").hex(), "gid":dev.local_gid.hex()}

def server():
  dev = BNXTDev(PCIDevice("bnxt", os.getenv("BNXT_PCI", "0000:41:00.0")), ip=os.getenv("BNXT_IP", REMOTE_IP))
  qp = BNXTQP(dev)
  print(json.dumps(qp_info(dev, qp)), flush=True)

  peer = json.loads(sys.stdin.readline())
  qp.connect(peer["qpn"], bytes.fromhex(peer["gid"]), int(peer["mac"], 16))
  print("connected", flush=True)

  target, target_paddrs = dev.pci_dev.alloc_sysmem(0x1000)
  target[:0x1000] = bytes(0x1000)
  rkey = dev.register_mem(target_paddrs, 0x1000)
  print(json.dumps({"target_addr":target_paddrs[0], "rkey":rkey}), flush=True)

  assert sys.stdin.readline().strip() == "done"
  received = bytes(target).rstrip(b"\0")
  print(f"AS TEXT: {received.decode(errors='replace')!r}", flush=True)
  print(json.dumps({"data":received.hex()}), flush=True)

def sync_remote():
  if os.getenv("SYNC", "1") == "0": return
  print("syncing BNXT driver to remote")
  subprocess.run(["rsync", "-azR", *SYNC_FILES, f"{REMOTE}:~/tinygrad/"], cwd=TINYGRAD, check=True)

def start_remote() -> subprocess.Popen[str]:
  print("booting remote")
  command = (f"cd ~/tinygrad && sudo env PYTHONPATH=. PYTHONUNBUFFERED=1 BNXT_DEBUG={os.getenv('BNXT_DEBUG', '0')} "
             f"BNXT_PCI={REMOTE_PCI} BNXT_IP={REMOTE_IP} python3 extra/bnxt_driver/connect.py --server")
  return subprocess.Popen(SSH + [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)

def client():
  assert 0 < len(MESSAGE) <= 0x1000
  sync_remote()
  remote = start_remote()
  assert remote.stdin is not None and remote.stdout is not None
  remote_info = read_json(remote.stdout, "QP information")
  print("booting local")
  dev = BNXTDev(PCIDevice("bnxt", LOCAL_PCI), ip=LOCAL_IP)
  qp = BNXTQP(dev)

  send_line(remote.stdin, qp_info(dev, qp))
  wait_line(remote.stdout, "connected")
  qp.connect(remote_info["qpn"], bytes.fromhex(remote_info["gid"]), int(remote_info["mac"], 16))
  print("both QPs in RTS")

  remote_target = read_json(remote.stdout, "MR information")
  source, source_paddrs = dev.pci_dev.alloc_sysmem(0x1000)
  source[:len(MESSAGE)] = MESSAGE
  lkey = dev.register_mem(source_paddrs, 0x1000)
  print(f"RDMA WRITE {len(MESSAGE)}B to remote phys 0x{remote_target['target_addr']:x}")
  qp.rdma_write(remote_target["target_addr"], remote_target["rkey"], source_paddrs[0], lkey, len(MESSAGE))

  send_line(remote.stdin, "done")
  wait_line(remote.stdout, "AS TEXT")
  result = read_json(remote.stdout, "RDMA result")
  assert bytes.fromhex(result["data"]) == MESSAGE
  print("RDMA WRITE data verified")

  remote.stdin.close()
  assert remote.wait() == 0
  print("RDMA WRITE test complete")

if __name__ == "__main__":
  server() if "--server" in sys.argv else client()
