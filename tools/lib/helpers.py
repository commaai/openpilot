import bz2
import datetime

TIME_FMT = "%Y-%m-%d--%H-%M-%S"

# regex patterns
class RE:
  DONGLE_ID =  r'(?P<dongle_id>[a-z0-9]{16})'
  TIMESTAMP = r'(?P<timestamp>[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2})'
  LOG_ID_V2 = r'(?P<count>[a-z0-9]{8})--(?P<uid>[a-z0-9]{10})'
  LOG_ID = r'(?P<log_id>(?:{}|{}))'.format(TIMESTAMP, LOG_ID_V2)
  ROUTE_NAME = r'(?P<route_name>{}[|_/]{})'.format(DONGLE_ID, LOG_ID)
  SEGMENT_NAME = r'{}(?:--|/)(?P<segment_num>[0-9]+)'.format(ROUTE_NAME)

  INDEX = r'-?[0-9]+'
  SLICE = r'(?P<start>{})?:?(?P<end>{})?:?(?P<step>{})?'.format(INDEX, INDEX, INDEX)
  SEGMENT_RANGE = r'{}(?:--|/)?(?P<slice>({}))?/?(?P<selector>([qras]))?'.format(ROUTE_NAME, SLICE)

  BOOTLOG_NAME = ROUTE_NAME

  EXPLORER_FILE = r'^(?P<segment_name>{})--(?P<file_name>[a-z]+\.[a-z0-9]+)$'.format(SEGMENT_NAME)
  OP_SEGMENT_DIR = r'^(?P<segment_name>{})$'.format(SEGMENT_NAME)


def timestamp_to_datetime(t: str) -> datetime.datetime:
  """
    Convert an openpilot route timestamp to a python datetime
  """
  return datetime.datetime.strptime(t, TIME_FMT)


def save_log(dest, log_msgs, compress=True):
  dat = b"".join(msg.as_builder().to_bytes() for msg in log_msgs)

  if compress:
    dat = bz2.compress(dat)

  with open(dest, "wb") as f:
    f.write(dat)
