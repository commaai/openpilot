#!/usr/bin/env python3
import os, sys, time
from tinygrad.runtime.support.system import RemotePCIDevice

LAT_N_RUNS = 500
THROUGHPUT_N_RUNS = 8
SIZES = [4, 1 << 10, 8 << 20]

if __name__ == "__main__":
  os.environ["REMOTE"] = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REMOTE", "127.0.0.1:6667")

  # choose any amd/nv gpu.
  devs = RemotePCIDevice.remote_list(0x1002, ((0, (0,)),), 0) or RemotePCIDevice.remote_list(0x10de, ((0, (0,)),), 0x03)
  if not devs: raise RuntimeError("no GPU found on remote")

  sock, name = devs[0]
  pci = RemotePCIDevice("BN", name, sock=sock)
  print(f"connected to {os.environ['REMOTE']}, device: {name}\n")

  # ping (minimal server round-trip, no device I/O)
  from tinygrad.runtime.support.system import RemoteCmd
  sock = pci.sock
  for _ in range(10): RemotePCIDevice._rpc(sock, 0, RemoteCmd.PING)
  st = time.perf_counter()
  for _ in range(LAT_N_RUNS): RemotePCIDevice._rpc(sock, 0, RemoteCmd.PING)
  ping_lat = (time.perf_counter() - st) / LAT_N_RUNS
  print(f"PING latency: {ping_lat*1e6:.1f} us ({1/ping_lat:,.0f} ops/sec)\n")

  # throughput
  sysmem, _ = pci.alloc_sysmem(max(SIZES))
  print(f"{'size':>10s}  {'write MB/s':>10s}  {'read MB/s':>10s}")
  for sz in SIZES:
    data = b'\x01' * sz

    for _ in range(5): sysmem[0:sz] = data
    st = time.perf_counter()
    for _ in range(THROUGHPUT_N_RUNS): sysmem[0:sz] = data
    pci.read_config(0, 4) # flush, since writes are posted
    w = (time.perf_counter() - st) / THROUGHPUT_N_RUNS

    for _ in range(5): sysmem[0:sz]
    st = time.perf_counter()
    for _ in range(THROUGHPUT_N_RUNS): sysmem[0:sz]
    r = (time.perf_counter() - st) / THROUGHPUT_N_RUNS

    sfx, div = [('B',1),('K',1<<10),('M',1<<20)][[sz>=1<<10,sz>=1<<20,sz>=1<<30].count(True)]
    print(f"{sz/div:>9.4g}{sfx}  {sz/w/1e6:>10.1f}  {sz/r/1e6:>10.1f}")
