#!/usr/bin/env python3
import subprocess, json, sys, os

REMOTE_HOST = os.getenv("REMOTE_HOST", "192.168.52.154")
LOCAL_PCI   = os.getenv("MLX_PCI", "0000:41:00.0")
REMOTE_PCI  = os.getenv("REMOTE_PCI", "0000:41:00.0")
LOCAL_IP    = os.getenv("LOCAL_IP", "10.0.0.1")
REMOTE_IP   = os.getenv("REMOTE_IP", "10.0.0.2")
SSH         = ["ssh", "-o", "StrictHostKeyChecking=no", REMOTE_HOST]
TINYGRAD    = os.path.dirname(os.path.abspath(__file__)) + "/../.."

print("syncing code to remote")
subprocess.run(["rsync", "-az", "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
                TINYGRAD + "/", f"{REMOTE_HOST}:~/tinygrad/"], check=True)

print("booting remote")
remote = subprocess.Popen(
  SSH + [f"cd ~/tinygrad && sudo PYTHONPATH=. MLX_DEBUG=1 MLX_PCI={REMOTE_PCI} MLX_IP={REMOTE_IP} python3 extra/mlx_driver/mlxdev.py --server"],
  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)

remote_info = None
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  try: remote_info = json.loads(line.strip()); break
  except json.JSONDecodeError: pass
assert remote_info, "failed to get remote connection info"

print("booting local")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from extra.mlx_driver.mlxdev import MLXDev, MLXQP
from tinygrad.runtime.support.system import PCIDevice

local_dev = MLXDev(PCIDevice("mlx5", LOCAL_PCI), ip=LOCAL_IP)
local_qp = MLXQP(local_dev)
local_info = {"qpn": local_qp.qpn, "mac": local_dev.mac.to_bytes(6,'big').hex(), "gid": local_dev.local_gid.hex()}

remote.stdin.write(json.dumps(local_info) + "\n")
remote.stdin.flush()
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  if "connected" in line: break

local_qp.connect(remote_info["qpn"], int(remote_info["mac"], 16), int(remote_info["gid"], 16))
print("both QPs in RTS")

remote_target = None
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  try: remote_target = json.loads(line.strip()); break
  except json.JSONDecodeError: pass
assert remote_target

test_msg = b"Test message, rdma works!"
src_mem, src_paddrs = local_dev.pci_dev.alloc_sysmem(0x1000)
for i, b in enumerate(test_msg): src_mem[i] = b

print(f"RDMA WRITE {len(test_msg)}B to remote phys 0x{remote_target['target_addr']:x}")
local_qp.rdma_write(remote_target["target_addr"], remote_target["rkey"], src_paddrs[0], local_dev.mkey, len(test_msg))

remote.stdin.write("done\n")
remote.stdin.flush()
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  if "AS TEXT" in line: break

remote.stdin.close()
remote.wait()
print("RDMA WRITE test complete")
