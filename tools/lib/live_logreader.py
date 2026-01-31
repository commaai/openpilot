import os
from cereal import log as capnp_log, messaging
from cereal.services import SERVICE_LIST

from openpilot.tools.lib.logreader import LogIterable, RawLogIterable


ALL_SERVICES = list(SERVICE_LIST.keys())

def raw_live_logreader(services: list[str] = ALL_SERVICES, addr: str = '127.0.0.1') -> RawLogIterable:
  if addr != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.reset_context()

  poller = messaging.Poller()

  for m in services:
    messaging.sub_sock(m, poller, addr=addr)

  while True:
    polld = poller.poll(100)
    for sock in polld:
      msg = sock.receive()
      yield msg


def live_logreader(services: list[str] = ALL_SERVICES, addr: str = '127.0.0.1') -> LogIterable:
  for m in raw_live_logreader(services, addr):
    with capnp_log.Event.from_bytes(m) as evt:
      yield evt
