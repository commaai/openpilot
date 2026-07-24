#!/usr/bin/env python3
import socket, struct, sys
from tinygrad.runtime.support.system import PCIDevice, RemoteCmd, System
from tinygrad.helpers import DEBUG, OSX

def resp(resp0=0, resp1=0, status=0): return struct.pack('<BQQ', status, resp0, resp1)
def resp_err(msg): return struct.pack('<BQQ', 1, len(err:=msg.encode()), 0) + err

discovered_devices: list[str] = []
opened_devices: dict[int, PCIDevice] = {}
mapped_bars: dict[tuple[int, int], object] = {}
sysmem_allocs: list[tuple] = []

def handle(conn, cmd, dev_id, bar, arg0, arg1, arg2):
  if cmd == RemoteCmd.PING:
    return conn.sendall(resp())

  if cmd == RemoteCmd.PROBE:
    payload = conn.recv(arg1, socket.MSG_WAITALL) if arg1 > 0 else b""
    filter_devices: dict[int, list[int]] = {}
    for i in range(0, len(payload), 8):
      mask, dev = struct.unpack('<II', payload[i:i+8])
      filter_devices.setdefault(mask, []).append(dev)
    base_class = None if arg0 == 0 else int(arg0)
    devs = System.list_devices(arg2, tuple([(x, tuple(y)) for x,y in filter_devices.items()]), base_class)
    for p in devs:
      if p not in discovered_devices: discovered_devices.append(p)
    data = "\n".join(f"{p[1]}:{discovered_devices.index(p)}" for p in devs).encode()
    return conn.sendall(resp(len(data), len(devs)) + data)

  # lazy device open
  if dev_id not in opened_devices:
    if dev_id >= len(discovered_devices): raise RuntimeError(f"device {dev_id} not probed")
    cl, pcibus = discovered_devices[dev_id]
    opened_devices[dev_id] = cl("SV", pcibus)
  pci_dev = opened_devices[dev_id]

  if cmd == RemoteCmd.MAP_BAR:
    if (dev_id, bar) not in mapped_bars: mapped_bars[(dev_id, bar)] = pci_dev.map_bar(bar)
    conn.sendall(resp(*pci_dev.bar_info(bar)))
  elif cmd == RemoteCmd.CFG_READ:
    conn.sendall(resp(pci_dev.read_config(arg0, arg1)))
  elif cmd == RemoteCmd.CFG_WRITE:
    pci_dev.write_config(arg0, arg2, arg1)
    conn.sendall(resp())
  elif cmd == RemoteCmd.RESIZE_BAR:
    pci_dev.resize_bar(bar)
    conn.sendall(resp())
  elif cmd == RemoteCmd.RESET:
    pci_dev.reset()
    conn.sendall(resp())
  elif cmd == RemoteCmd.MMIO_READ:
    bar_view = mapped_bars[(dev_id, bar)]
    if arg0 % 4 == 0 and arg1 == 4: conn.sendmsg([resp(arg1), struct.pack('<I', bar_view.view(fmt='I')[arg0 // 4])])
    else: conn.sendmsg([resp(arg1), bar_view[arg0:arg0+arg1]])
  elif cmd == RemoteCmd.MMIO_WRITE:
    data = conn.recv(arg1, socket.MSG_WAITALL)
    bar_view = mapped_bars[(dev_id, bar)]
    if arg0 % 4 == 0 and arg1 == 4: bar_view.view(fmt='I')[arg0 // 4] = struct.unpack('<I', data)[0]
    else: bar_view[arg0:arg0+arg1] = data
  elif cmd == RemoteCmd.MAP_SYSMEM:
    memview, paddrs = pci_dev.alloc_sysmem(arg0, contiguous=bool(arg1))
    sysmem_allocs.append((memview, paddrs))
    paddrs_bytes = struct.pack(f'<{len(paddrs)}Q', *paddrs)
    conn.sendall(resp(len(paddrs_bytes), len(sysmem_allocs) - 1) + paddrs_bytes)
  elif cmd == RemoteCmd.SYSMEM_READ:
    conn.sendmsg([resp(arg1), sysmem_allocs[bar][0][arg0:arg0+arg1]])
  elif cmd == RemoteCmd.SYSMEM_WRITE:
    sysmem_allocs[bar][0][arg0:arg0+arg1] = conn.recv(arg1, socket.MSG_WAITALL)
  else: raise RuntimeError(f"unknown command {cmd}")

def serve(conn:socket.socket):
  REQ = '<BIIQQQ'
  while True:
    hdr = conn.recv(struct.calcsize(REQ), socket.MSG_WAITALL)
    if len(hdr) < struct.calcsize(REQ): raise ConnectionError("client disconnected")
    cmd, dev_id, bar, arg0, arg1, arg2 = struct.unpack(REQ, hdr)
    if DEBUG >= 4: print(f"cmd={RemoteCmd(cmd).name} dev={dev_id} bar={bar} arg0={arg0:#x} arg1={arg1:#x} arg2={arg2:#x}")
    try: handle(conn, cmd, dev_id, bar, arg0, arg1, arg2)
    except ConnectionError: raise
    except Exception as e:
      if cmd in {RemoteCmd.MMIO_WRITE, RemoteCmd.SYSMEM_WRITE}: raise ConnectionError(f"write failed: {e}")
      print(f"ERROR: {e}")
      conn.sendall(resp_err(str(e)))

if __name__ == "__main__":
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 6667
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server.bind(("0.0.0.0", port))
  server.listen(1)
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try: s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
  finally: s.close()
  print(f"listening on {ip}:{port}")
  while True:
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for bt in [socket.SO_SNDBUF, socket.SO_RCVBUF]: conn.setsockopt(socket.SOL_SOCKET, bt, 64 << 20)
    try: serve(conn)
    except ConnectionError: print("disconnected")
