import cereal
import json
import copy
from operator import attrgetter
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
from base64 import b64encode

class Base64Encoder(json.JSONEncoder):
    # pylint: disable=method-hidden
    def default(self, o):
        if isinstance(o, bytes):
            return b64encode(o).decode()
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
    typeToChannel[k] = channel_id = writer.register_channel(
      topic=k,
      message_encoding=MessageEncoding.JSON,
      schema_id=type_to_schema[k],
    )
  logf = open('example.qlog', 'rb')
  events = cereal.log.Event.read_multiple(logf)
  for event in events:
    e = event.to_dict()
    writer.add_message(
        typeToChannel[str(event.which)],
        log_time=int(e["logMonoTime"]),
        data=json.dumps(e, cls=Base64Encoder).encode("utf-8"),
        publish_time=int(e["logMonoTime"]),
    )

  writer.finish()
