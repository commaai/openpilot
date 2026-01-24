import time
from cereal import log

NO_TRAVERSAL_LIMIT = 2**64 - 1


def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  """Convert list of CAN messages to Cap'n Proto serialized bytes.

  Args:
    can_msgs: List of tuples [(address, data_bytes, src), ...]
    msgtype: 'can' or 'sendcan'
    valid: Whether the event is valid

  Returns:
    Cap'n Proto serialized bytes
  """
  dat = log.Event.new_message(
    valid=valid,
    logMonoTime=int(time.monotonic() * 1e9)
  )

  can_data = dat.init(msgtype, len(can_msgs))
  for i, msg in enumerate(can_msgs):
    can_data[i].address = msg[0]
    can_data[i].dat = msg[1]
    can_data[i].src = msg[2]

  return dat.to_bytes()


def can_capnp_to_list(strings, msgtype='can'):
  """Convert Cap'n Proto serialized bytes to list of CAN messages.

  Args:
    strings: Tuple/list of serialized Cap'n Proto bytes
    msgtype: 'can' or 'sendcan'

  Returns:
    List of tuples [(nanos, [(address, data, src), ...]), ...]
  """
  result = []
  for s in strings:
    with log.Event.from_bytes(s, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as event:
      frames = getattr(event, msgtype)
      frame_list = [(f.address, f.dat, f.src) for f in frames]
      result.append((event.logMonoTime, frame_list))
  return result
