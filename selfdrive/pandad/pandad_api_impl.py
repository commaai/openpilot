import time
from cereal import log

NO_TRAVERSAL_LIMIT = 2**64 - 1

# Cache schema fields for faster access (avoids string lookup on each field access)
_cached_reader_fields = None  # (address_field, dat_field, src_field) for reading
_cached_writer_fields = None  # (address_field, dat_field, src_field) for writing


def _get_reader_fields(schema):
  """Get cached schema field objects for reading."""
  global _cached_reader_fields
  if _cached_reader_fields is None:
    fields = schema.fields
    _cached_reader_fields = (fields['address'], fields['dat'], fields['src'])
  return _cached_reader_fields


def _get_writer_fields(schema):
  """Get cached schema field objects for writing."""
  global _cached_writer_fields
  if _cached_writer_fields is None:
    fields = schema.fields
    _cached_writer_fields = (fields['address'], fields['dat'], fields['src'])
  return _cached_writer_fields


def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  """Convert list of CAN messages to Cap'n Proto serialized bytes.

  Args:
    can_msgs: List of tuples [(address, data_bytes, src), ...]
    msgtype: 'can' or 'sendcan'
    valid: Whether the event is valid

  Returns:
    Cap'n Proto serialized bytes
  """
  global _cached_writer_fields

  dat = log.Event.new_message(valid=valid, logMonoTime=int(time.monotonic() * 1e9))
  can_data = dat.init(msgtype, len(can_msgs))

  # Cache schema fields on first call
  if _cached_writer_fields is None and len(can_msgs) > 0:
    _cached_writer_fields = _get_writer_fields(can_data[0].schema)

  if _cached_writer_fields is not None:
    addr_f, dat_f, src_f = _cached_writer_fields
    for i, msg in enumerate(can_msgs):
      f = can_data[i]
      f._set_by_field(addr_f, msg[0])
      f._set_by_field(dat_f, msg[1])
      f._set_by_field(src_f, msg[2])

  return dat.to_bytes()


def can_capnp_to_list(strings, msgtype='can'):
  """Convert Cap'n Proto serialized bytes to list of CAN messages.

  Args:
    strings: Tuple/list of serialized Cap'n Proto bytes
    msgtype: 'can' or 'sendcan'

  Returns:
    List of tuples [(nanos, [(address, data, src), ...]), ...]
  """
  global _cached_reader_fields
  result = []

  for s in strings:
    with log.Event.from_bytes(s, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as event:
      frames = getattr(event, msgtype)

      # Cache schema fields on first frame for faster access
      if _cached_reader_fields is None and len(frames) > 0:
        _cached_reader_fields = _get_reader_fields(frames[0].schema)

      if _cached_reader_fields is not None:
        addr_f, dat_f, src_f = _cached_reader_fields
        frame_list = [(f._get_by_field(addr_f), f._get_by_field(dat_f), f._get_by_field(src_f)) for f in frames]
      else:
        frame_list = []

      result.append((event.logMonoTime, frame_list))
  return result
