#!/usr/bin/env python3
"""CPU-driven SEND/RECV between hosts. Sync the tree first and run with PYTHONPATH=."""
import atexit, json, shlex, subprocess, sys, time
from tinygrad.helpers import getenv
from tinygrad.runtime.support.rdma.bnxtdev import BNXTDev, BNXTQP
from tinygrad.runtime.support.system import PCIDevice

SIZE, ITERS = getenv("SIZE", 0x2000), getenv("ITERS", 257)

def read_json(stream):
  for line in stream:
    try: return json.loads(line)
    except json.JSONDecodeError: print(line, end="")
  raise RuntimeError("peer disconnected")

def write_json(stream, obj):
  stream.write(json.dumps(obj) + "\n")
  stream.flush()

def endpoint():
  dev = BNXTDev(PCIDevice("bnxt", getenv("BNXT_PCI", "0000:41:00.0")))
  atexit.register(dev.fini)
  mem, pages = dev.pci_dev.alloc_sysmem(SIZE)
  return dev, BNXTQP(dev), mem, pages[0], dev.register_mem(pages, SIZE)

def info(dev, qp): return {"qpn":qp.qpn, "gid":dev.local_gid.hex(), "mac":dev.mac}
def connect(qp, peer): qp.connect(peer["qpn"], bytes.fromhex(peer["gid"]), peer["mac"])

def server():
  dev, qp, mem, addr, key = endpoint()
  write_json(sys.stdout, info(dev, qp))
  connect(qp, read_json(sys.stdin))
  for i in range(ITERS):
    mem[:] = bytes(SIZE)
    qp.post_recv(addr, key, SIZE)
    qp.poll(qp.rcq, qp.rcq_id)
    assert bytes(mem[:]) == bytes([i % 255 + 1]) * SIZE, f"receive mismatch at iteration {i}"
  write_json(sys.stdout, {"received":ITERS})

if __name__ == "__main__":
  if "--server" in sys.argv: server()
  else:
    env = {"PYTHONPATH":".", "BNXT_PCI":getenv("REMOTE_PCI", "0000:41:00.0"), "SIZE":str(SIZE), "ITERS":str(ITERS)}
    command = f"cd {shlex.quote(getenv('REMOTE_DIR', 'tinygrad'))} && " + shlex.join([
      "env", *(f"{k}={v}" for k, v in env.items()), "python3", "-u", "extra/bnxt_driver/connect.py", "--server"])
    with subprocess.Popen(["ssh", "-o", "BatchMode=yes", getenv("REMOTE_HOST", "192.168.52.213"), command],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as remote:
      peer = read_json(remote.stdout)
      dev, qp, mem, addr, key = endpoint()
      write_json(remote.stdin, info(dev, qp))
      connect(qp, peer)
      start = time.perf_counter()
      for i in range(ITERS):
        mem[:] = bytes([i % 255 + 1]) * SIZE
        qp.post_send(addr, key, SIZE)
        qp.poll(qp.scq, qp.scq_id)
      assert read_json(remote.stdout) == {"received":ITERS}
      assert remote.wait() == 0
      print(f"BNXT SEND/RECV passed: {ITERS} x {SIZE} bytes, {SIZE * ITERS / (time.perf_counter() - start) / 1e9:.3f} GB/s")
