from openpilot.cereal import log as capnp_log, messaging
from openpilot.cereal.services import SERVICE_LIST

from openpilot.tools.lib.logreader import LogIterable, RawLogIterable


ALL_SERVICES = list(SERVICE_LIST.keys())

def raw_live_logreader(services: list[str] = ALL_SERVICES) -> RawLogIterable:
  poller = messaging.Poller()

  for m in services:
    messaging.sub_sock(m, poller)

  while True:
    polld = poller.poll(100)
    for sock in polld:
      msg = sock.receive()
      yield msg


def live_logreader(services: list[str] = ALL_SERVICES) -> LogIterable:
  for m in raw_live_logreader(services):
    with capnp_log.Event.from_bytes(m) as evt:
      yield evt
