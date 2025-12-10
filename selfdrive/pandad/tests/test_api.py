
import pytest
from openpilot.selfdrive.pandad import can_list_to_can_capnp, can_capnp_to_list

def test_round_trip():
  # [(addr, data, src)]
  msgs = [
    [123, b'data123', 0],
    [456, b'data456', 1]
  ]

  capnp_out = can_list_to_can_capnp(msgs, msgtype='can')
  assert len(capnp_out) > 0

  decoded = can_capnp_to_list([capnp_out], msgtype='can')
  # Structure: [(nanos, [(addr, data, src)])]
  assert len(decoded) == 1
  nanos, frames = decoded[0]
  assert len(frames) == 2
  assert frames[0] == (123, b'data123', 0)
  assert frames[1] == (456, b'data456', 1)

def test_sendcan():
  msgs = [[0x200, b'mydata', 128]]
  capnp_out = can_list_to_can_capnp(msgs, msgtype='sendcan')
  decoded = can_capnp_to_list([capnp_out], msgtype='sendcan')
  assert decoded[0][1][0] == (0x200, b'mydata', 128)

def test_errors():
   with pytest.raises(TypeError):
     can_list_to_can_capnp("not a list")
   with pytest.raises(ValueError):
     can_list_to_can_capnp([[123, b'd']])
   with pytest.raises(TypeError):
     can_list_to_can_capnp([[123, "string_not_bytes", 0]])
