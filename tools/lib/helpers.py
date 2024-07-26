# regex patterns
class RE:
  DONGLE_ID = r'(?P<dongle_id>[a-f0-9]{16})'
  TIMESTAMP = r'(?P<timestamp>[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2})'
  LOG_ID_V2 = r'(?P<count>[a-f0-9]{8})--(?P<uid>[a-z0-9]{10})'
  LOG_ID = fr'(?P<log_id>(?:{TIMESTAMP}|{LOG_ID_V2}))'
  ROUTE_NAME = fr'(?P<route_name>{DONGLE_ID}[|_/]{LOG_ID})'
  SEGMENT_NAME = fr'{ROUTE_NAME}(?:--|/)(?P<segment_num>[0-9]+)'

  INDEX = r'-?[0-9]+'
  SLICE = fr'(?P<start>{INDEX})?:?(?P<end>{INDEX})?:?(?P<step>{INDEX})?'
  SEGMENT_RANGE = fr'{ROUTE_NAME}(?:(--|/)(?P<slice>({SLICE})))?(?:/(?P<selector>([qras])))?'

  BOOTLOG_NAME = ROUTE_NAME

  EXPLORER_FILE = fr'^(?P<segment_name>{SEGMENT_NAME})--(?P<file_name>[a-z]+\.[a-z0-9]+)$'
  OP_SEGMENT_DIR = fr'^(?P<segment_name>{SEGMENT_NAME})$'
