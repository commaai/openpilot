import bz2
import datetime

TIME_FMT = "%Y-%m-%d--%H-%M-%S"


# regex patterns
class RE:
  DONGLE_ID_BASE = '[a-f0-9]{16}'
  DONGLE_ID = r'(?P<dongle_id>{})'.format(DONGLE_ID_BASE)
  TIMESTAMP = r'(?P<timestamp>[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2})'
  LOG_ID_BASE = r'[a-z0-9-]{20}'
  LOG_ID = r'(?P<log_id>{})'.format(LOG_ID_BASE)
  ROUTE_NAME = r'(?P<route_name>{}[|_/]{})'.format(DONGLE_ID, LOG_ID)
  SEGMENT_NUMBER_BASE = r'[0-9]+'
  SEGMENT_NUMBER = r'(?P<segment_num>{})'.format(SEGMENT_NUMBER_BASE)
  SEGMENT_NAME = r'{}(?:--|/){}'.format(ROUTE_NAME, SEGMENT_NUMBER)
  FILE_NAME_BASE = r'([fde]?camera.hevc|[qr]?log.bz2)'
  FILE_NAME = r'(?P<file_name>{})'.format(FILE_NAME_BASE)

  INDEX = r'-?[0-9]+'
  SLICE = r'(?P<start>{})?:?(?P<end>{})?:?(?P<step>{})?'.format(INDEX, INDEX, INDEX)
  SEGMENT_RANGE = r'{}(?:(--|/)(?P<slice>({})))?(?:/(?P<selector>([qras])))?'.format(ROUTE_NAME, SLICE)

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
