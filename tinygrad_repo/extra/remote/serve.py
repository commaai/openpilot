#!/usr/bin/env python3
import socket, struct, sys, signal
from tinygrad.runtime.support.system import PCIDevice, RemoteCmd, System, REMOTE_REQ, REMOTE_RESP
from tinygrad.runtime.support.am.amdev import AMMemoryManager
from tinygrad.runtime.support.system import FileIOInterface
from tinygrad.device import Device, TinyELF
from tinygrad.helpers import DEBUG, Target, to_mv

def resp(resp0=0, resp1=0, status=0): return struct.pack(REMOTE_RESP, status, resp0, resp1)
def resp_err(msg): return resp(len(err:=msg.encode()), status=1) + err

discovered_devices: list[tuple[type, str]] = []
opened_devices: dict[int, PCIDevice] = {}
mapped_bars: dict[tuple[int, int], object] = {}
programs: list = []

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

  # only PCI commands need an open GPU
  if cmd not in {RemoteCmd.MAP_SYSMEM, RemoteCmd.SYSMEM_READ, RemoteCmd.SYSMEM_WRITE, RemoteCmd.UNMAP_SYSMEM, RemoteCmd.LOAD_PROG, RemoteCmd.EXEC_PROG}:
    if dev_id not in opened_devices:
      if dev_id >= len(discovered_devices): raise RuntimeError(f"device {dev_id} not probed")
      cl, pcibus = discovered_devices[dev_id]
      opened_devices[dev_id] = cl("SV", pcibus)
    pci_dev = opened_devices[dev_id]

  if cmd == RemoteCmd.MAP_BAR:
    if (dev_id, bar) not in mapped_bars: mapped_bars[(dev_id, bar)] = pci_dev.map_bar(bar)
    conn.sendall(resp(*pci_dev.bar_info(bar)) + struct.pack('<Q', mapped_bars[(dev_id, bar)].addr))
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
    elif arg0 % 8 == 0 and arg1 == 8: conn.sendmsg([resp(arg1), struct.pack('<Q', bar_view.view(fmt='Q')[arg0 // 8])])
    else: conn.sendmsg([resp(arg1), bar_view[arg0:arg0+arg1]])
  elif cmd == RemoteCmd.MMIO_WRITE:
    data = conn.recv(arg1, socket.MSG_WAITALL)
    bar_view = mapped_bars[(dev_id, bar)]
    if arg0 % 4 == 0 and arg1 == 4: bar_view.view(fmt='I')[arg0 // 4] = struct.unpack('<I', data)[0]
    elif arg0 % 8 == 0 and arg1 == 8: bar_view.view(fmt='Q')[arg0 // 8] = struct.unpack('<Q', data)[0]
    else: bar_view[arg0:arg0+arg1] = data
  elif cmd == RemoteCmd.MAP_SYSMEM:
    memview, paddrs = System.alloc_sysmem(arg0, vaddr=arg2, contiguous=bool(arg1))
    paddrs_bytes = struct.pack(f'<{len(paddrs) + 1}Q', memview.addr, *paddrs)
    conn.sendall(resp(len(paddrs_bytes)) + paddrs_bytes)
  elif cmd == RemoteCmd.SYSMEM_READ:
    conn.sendmsg([resp(arg1), to_mv(arg0, arg1)])
  elif cmd == RemoteCmd.SYSMEM_WRITE:
    to_mv(arg0, arg1)[:] = conn.recv(arg1, socket.MSG_WAITALL)
  elif cmd == RemoteCmd.UNMAP_SYSMEM:
    FileIOInterface.munmap(arg0, arg1)
    conn.sendall(resp())
  elif cmd == RemoteCmd.LOAD_PROG:
    programs.append(Device["CPU"].runtime(TinyELF(conn.recv(arg0, socket.MSG_WAITALL), "hcq_submit", Target("CPU"), ())))
    conn.sendall(resp(len(programs) - 1))
  elif cmd == RemoteCmd.EXEC_PROG:
    et = programs[arg0](*struct.unpack(f'<{arg1}Q', conn.recv(arg1 * 8, socket.MSG_WAITALL)), wait=bool(arg2))
    if (mock:=sys.modules.get("test.mockgpu.mockgpu")) is not None: # native programs bypass the mock's memoryview hooks
      for d in mock.drivers: d._emulate_execute()
    if arg2: conn.sendall(resp(int(et * 1e9)))
  else: raise RuntimeError(f"unknown command {cmd}")

def serve(conn:socket.socket):
  while True:
    hdr = conn.recv(struct.calcsize(REMOTE_REQ), socket.MSG_WAITALL)
    if len(hdr) < struct.calcsize(REMOTE_REQ): raise ConnectionError("client disconnected")
    cmd, dev_id, bar, arg0, arg1, arg2 = struct.unpack(REMOTE_REQ, hdr)
    if DEBUG >= 4: print(f"cmd={RemoteCmd(cmd).name} dev={dev_id} bar={bar} arg0={arg0:#x} arg1={arg1:#x} arg2={arg2:#x}")
    try: handle(conn, cmd, dev_id, bar, arg0, arg1, arg2)
    except ConnectionError: raise
    except Exception as e:
      if cmd in {RemoteCmd.MMIO_WRITE, RemoteCmd.SYSMEM_WRITE} or (cmd == RemoteCmd.EXEC_PROG and not arg2):
        raise ConnectionError(f"posted command failed: {e}")
      print(f"ERROR: {e}")
      conn.sendall(resp_err(str(e)))

if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
  System.reserve_va(AMMemoryManager.va_allocator.base, AMMemoryManager.va_allocator.size)
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 6667
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server.bind(("0.0.0.0", port))
  server.listen(1)
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try: s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
  finally: s.close()
  print(f"listening on {ip}:{port}", flush=True)
  while True:
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for bt in [socket.SO_SNDBUF, socket.SO_RCVBUF]: conn.setsockopt(socket.SOL_SOCKET, bt, 64 << 20)
    try: serve(conn)
    except ConnectionError: print("disconnected")
    finally: conn.close()
