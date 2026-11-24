import asyncio


async def ssh_tunnel(host, key, ports, *, known_hosts=None):
  args = [
    "ssh", "-N", "-T", "-i", key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=5",
    "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2",
    "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "IPQoS=lowdelay",
  ]
  if known_hosts is not None:
    args.extend(["-o", f"UserKnownHostsFile={known_hosts}"])
  for port in ports:
    args.extend(["-L", f"127.0.0.1:{port}:127.0.0.1:8022"])
  while True:
    proc = await asyncio.create_subprocess_exec(*args, host)
    try:
      await proc.wait()
    finally:
      if proc.returncode is None:
        proc.terminate()
        await proc.wait()
    await asyncio.sleep(2)
