import time

from panda.tests.python.test_uds import UdsServicesType, UdsServer, MockCanBuffer, STANDARD_UDS_SERVER_SERVICES
from selfdrive.car.fw_versions import get_fw_versions
from panda.python import uds
from selfdrive.boardd.boardd import can_capnp_to_can_list, can_list_to_can_capnp
from cereal.messaging import log_from_bytes

TEST_UDS_SERVER_SERVICES: UdsServicesType = {
  uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER: {
    None: {  # no subfunction
      b'\xF1\x00': b'\xF1\x00CV1 MFC  AT USA LHD 1.00 1.05 99210-CV000 211027',
      b'\xF1\x90': b'\xF1\x901H3110W0RLD5',
      b'\xF1\x81': b'\xF1\x81\x018966306Q6000\x00\x00\x00\x00',
    }
  }
}
services = STANDARD_UDS_SERVER_SERVICES | TEST_UDS_SERVER_SERVICES

services[uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL][bytes([uds.SESSION_TYPE.DEFAULT])][b''] = b'\x00\x32\x01\xf4'
services[uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL][bytes([uds.SESSION_TYPE.EXTENDED_DIAGNOSTIC])][b''] = b'\x00\x32\x01\xf4'


# class to support interfacing with fake UDS server from IsoTpParallelQuery class
class MockCanSocket:  # TODO: rename to sock
  def __init__(self):
    self.timeout = 0.01  # delay, doesn't work without this. FIXME: why?
    self.can_buf = MockCanBuffer()

  def send(self, msgs: bytes, server: bool = False):
    # print('send', msgs)
    # print('event', can_capnp_to_can_list(log_from_bytes(msgs).sendcan))
    for msg in can_capnp_to_can_list(log_from_bytes(msgs).sendcan):
      self.can_buf.can_send(msg[0], msg[2], msg[3], server=server)
    # print('parsed', can_capnp_to_can_list([msgs]))
    # print('parsed', can_capnp_to_can_list([msgs]))
    # print('end')

  def receive(self, non_blocking=False):
    time.sleep(self.timeout)
    msgs = self.can_buf.can_recv()
    if len(msgs) == 0:
      return None
    return can_list_to_can_capnp(msgs, msgtype='can')


# can_buf = MockCanBuffer()
# can_buf.receive = can_buf.can_send

can_sock = MockCanSocket()

tx_addr = 0x700
sub_addr = None

# uds_server = UdsServer(can_buf, uds.get_rx_addr_for_tx_addr(tx_addr), tx_addr, sub_addr=sub_addr)
uds_server = UdsServer(can_sock.can_buf, uds.get_rx_addr_for_tx_addr(tx_addr), tx_addr, sub_addr=sub_addr)
uds_server.set_services(STANDARD_UDS_SERVER_SERVICES | TEST_UDS_SERVER_SERVICES)
uds_server.start()

try:
  # can_send = partial(can_buf.can_send(server=True))
  t = time.monotonic()
  fw_versions = get_fw_versions(can_sock, can_sock, query_brand='toyota')
  print('time', time.monotonic() - t)
  print('got fw versions', fw_versions)
  assert len(fw_versions) == 1 and fw_versions[0].fwVersion == b'\x018966306Q6000\x00\x00\x00\x00'
finally:
  uds_server.stop()
