import cereal
import json
import copy
from operator import attrgetter
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
from base64 import b64encode
import argparse
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentName
from openpilot.tools.lib.helpers import save_log
from urllib.parse import urlparse, parse_qs
import sys
import os
import tempfile
import multiprocessing
import math
import numbers
from openpilot.selfdrive.test.openpilotci import get_url

juggle_dir = os.path.dirname(os.path.realpath(__file__))

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"

def nan2None(obj):
    if isinstance(obj, dict):
        return {k:nan2None(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [nan2None(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

class Base64Encoder(json.JSONEncoder):
    def encode(self, obj, *args, **kwargs):
        return super().encode(nan2None(obj), *args, **kwargs)
    # pylint: disable=method-hidden
    def default(self, o):
        if isinstance(o, bytes):
            return b64encode(o).decode()
        if math.isnan(o):
            return 0
        return json.JSONEncoder.default(self, o)

typeMap = {
  "int8": {"type": "integer"},
  "int16": {"type": "integer"},
  "int32": {"type": "integer"},
  "int64": {"type": "integer"},
  "uint8": {"type": "integer"},
  "uint16": {"type": "integer"},
  "uint32": {"type": "integer"},
  "uint64": {"type": "string"},
  "float32": {"type": "number"},
  "float64": {"type": "number"},
  "text": {"type": "string"},
  "data": {"type": "string", "contentEncoding": "base64"},
  "bool": {"type": "boolean"},
}

def name_to_schema(name):
  file_name = name.split(':')[0].split('.')[0]
  type_name = name.split(':')[-1]
  path = f"{file_name}.{type_name}"
  return attrgetter(path)(cereal).schema



def list_schema_to_json(schema, et, bind = None):
  w = et.which
  if str(w) == "data":
    return { "type": "string" }
  elif str(w) in typeMap:
    return { "type": "array", "items": typeMap[str(w)] }
  elif str(w) == 'struct':
    name = schema.elementType.node.displayName
    try:
      field_schema = name_to_schema(name)
    except:
      print("skipping legacy data")
      return None
    return { "type": "array", "items": schema_to_json(field_schema, bind) }
  elif str(w) == 'enum':
    return { "type": "array", "items": { "type": "string", "enum": list(schema.elementType.enumerants.keys()) } }
  elif str(w) == 'list':
    return None

  else:
    print(f"warning, unsupported elementType: {et}")
    return { "type": "array" }

def schema_to_json(schema, bind = None):
  t = schema.node.which
  if t == 'struct':
    base = { "type": "object", "properties": {}}
    for f in schema.fields_list:
      w = f.proto.which
      if w == 'slot':
        ft = f.proto.slot.type.which
        if ft == 'struct':
          name = f.schema.node.displayName
          try:
            field_schema = name_to_schema(name)
            if f.schema.node.isGeneric:
              base["properties"][f.proto.name] = schema_to_json(field_schema, f.proto.slot.type.struct.brand.scopes[0].bind)
            else:
              base["properties"][f.proto.name] = schema_to_json(field_schema)
          except:
            print("skipping legacy data")
        elif str(ft) in typeMap:
          base["properties"][f.proto.name] = typeMap[str(ft)]
        elif str(ft) == 'list':
          et = f.proto.slot.type.list.elementType
          l = list_schema_to_json(f.schema, et, bind)
          if l is not None:
            base["properties"][f.proto.name] = l
          else:
            print("warning, foxglove does not support lists in lists, skipping field")
        elif str(ft) == 'enum':
          base["properties"][f.proto.name] = {"type": "string", "enum": list(f.schema.enumerants.keys())}
        elif str(ft) == 'anyPointer':
          bindIndex = f.proto.slot.type.anyPointer.parameter.parameterIndex
          pt = bind[bindIndex].type.which
          if str(pt) in typeMap:
            base["properties"][f.proto.name] = typeMap[str(pt)]
          else:
            print(f"warning, unsupported pointer type: {pt}")
        else:
          print(f"warning, unsupported schema type: {ft}")
      elif w == 'group':
        group = schema_to_json(f.schema)
        base = base | group
    return base
  else:
    print(f"warning, unsupported schema type: {t}")
    return None

def get_event_schemas():
  schemas = {}
  base_template = { "type": "object", "properties": { "logMonoTime": {"type": "string"}, "valid": {"type": "boolean"} } }
  for field in cereal.log.Event.schema.fields_list:
    base = copy.deepcopy(base_template)
    w = field.proto.which
    if field.proto.name not in cereal.log.Event.schema.union_fields:
      continue
    if w == 'slot':
      ft = field.proto.slot.type.which
      if ft == 'struct':
        name = field.schema.node.displayName
        try:
          field_schema = name_to_schema(name)
          if field.schema.node.isGeneric:
            base["properties"][field.proto.name] = schema_to_json(field_schema, field.proto.slot.type.struct.brand.scopes[0].bind)
          else:
            base["properties"][field.proto.name] = schema_to_json(field_schema)
        except:
          print("skipping legacy data")
      elif str(ft) in typeMap:
        base["properties"][field.proto.name] = typeMap[str(ft)]
      elif str(ft) == 'list':
        et = field.proto.slot.type.list.elementType
        l = list_schema_to_json(field.schema, et)
        if l is not None:
          base["properties"][field.proto.name] = l
        else:
          print("warning, foxglove does not support lists in lists, skipping field")
      elif str(ft) == 'enum':
        base["properties"][field.proto.name] = {"type": "string", "enum": list(field.schema.enumerants.keys())}
      else:
        print(f"warning, unsupported schema type: {ft}")
    schemas[field.proto.name] = base

  return schemas

schemas = get_event_schemas()

def juggle_route(route_or_segment_name, segment_count, qlog, ci=False):
  segment_start = 0
  if 'cabana' in route_or_segment_name:
    query = parse_qs(urlparse(route_or_segment_name).query)
    route_or_segment_name = query["route"][0]

  if route_or_segment_name.startswith(("http://", "https://", "cd:/")) or os.path.isfile(route_or_segment_name):
    logs = [route_or_segment_name]
  elif ci:
    route_or_segment_name = SegmentName(route_or_segment_name, allow_route_name=True)
    route = route_or_segment_name.route_name.canonical_name
    segment_start = max(route_or_segment_name.segment_num, 0)
    logs = [get_url(route, i) for i in range(100)]  # Assume there not more than 100 segments
  else:
    route_or_segment_name = SegmentName(route_or_segment_name, allow_route_name=True)
    segment_start = max(route_or_segment_name.segment_num, 0)

    if route_or_segment_name.segment_num != -1 and segment_count is None:
      segment_count = 1

    r = Route(route_or_segment_name.route_name.canonical_name, route_or_segment_name.data_dir)
    logs = r.qlog_paths() if qlog else r.log_paths()

  segment_end = segment_start + segment_count if segment_count else None
  logs = logs[segment_start:segment_end]

  if None in logs:
    resp = input(f"{logs.count(None)}/{len(logs)} of the rlogs in this segment are missing, would you like to fall back to the qlogs? (y/n) ")
    if resp == 'y':
      logs = r.qlog_paths()[segment_start:segment_end]
    else:
      print("Please try a different route or segment")
      return

  all_data = []
  with multiprocessing.Pool(24) as pool:
    for d in pool.map(load_segment, logs):
      all_data += d

  with tempfile.NamedTemporaryFile(suffix='.rlog', dir=juggle_dir) as tmp:
    save_log(tmp.name, all_data, compress=False)
    del all_data
    convert_log(tmp.name)

def load_segment(segment_name):
  if segment_name is None:
    return []

  try:
    return list(LogReader(segment_name))
  except (AssertionError, ValueError) as e:
    print(f"Error parsing {segment_name}: {e}")
    return []

def convert_log(log_file):
  channel_exclusions = ['logMonoTime', 'valid']
  with open("test.mcap", "wb") as f:
    writer = Writer(f)
    writer.start()

    type_to_schema = {}
    for k in list(schemas.keys()):
      schema_id = writer.register_schema(
        name=k,
        encoding=SchemaEncoding.JSONSchema,
        data=bytes(json.dumps(schemas[k]), "utf-8"),
      )
      type_to_schema[k] = schema_id

    typeToChannel = {}
    for k in list(schemas.keys()):
      if k in channel_exclusions:
        continue
      typeToChannel[k] = writer.register_channel(
        topic=k,
        message_encoding=MessageEncoding.JSON,
        schema_id=type_to_schema[k],
      )
    logf = open(log_file, 'rb')
    events = cereal.log.Event.read_multiple(logf)

    offset = 0
    for event in events:
      e = event.to_dict()
      if str(event.which) == "initData":
        offset = int(e["initData"]["wallTimeNanos"]) - int(e["logMonoTime"])
      writer.add_message(
          typeToChannel[str(event.which)],
          log_time=int(e["logMonoTime"]) + offset,
          data=json.dumps(e, cls=Base64Encoder).encode("utf-8"),
          publish_time=int(e["logMonoTime"]),
      )

    writer.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper to run PlotJuggler on openpilot routes",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("--qlog", action="store_true", help="Use qlogs")
  parser.add_argument("--ci", action="store_true", help="Download data from openpilot CI bucket")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to plot (cabana share URL accepted)")
  parser.add_argument("segment_count", type=int, nargs='?', help="The number of segments to plot")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  route_or_segment_name = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
  juggle_route(route_or_segment_name, args.segment_count, args.qlog, args.ci)



