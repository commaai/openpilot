import json
from mcap.well_known import SchemaEncoding, MessageEncoding
from base64 import b64encode
import math

def register_schema(writer, name, schema):
  return writer.register_schema(
    name=name,
    encoding=SchemaEncoding.JSONSchema,
    data=bytes(json.dumps(schema), "utf-8"),
  )

def register_channel(writer, topic, schema_id):
  return writer.register_channel(
    topic=topic,
    message_encoding=MessageEncoding.JSON,
    schema_id=schema_id,
  )

def register(writer, topic, schema_name, schema):
  schema_id = register_schema(writer, schema_name, schema)
  channel_id = register_channel(writer, topic, schema_id)
  return (schema_id, channel_id)

def message(writer, channel, event, offset, data):
  writer.add_message(
    channel,
    log_time=int(event["logMonoTime"]) + offset,
    data=json.dumps(data, cls=Encoder).encode("utf-8"),
    publish_time=int(event["logMonoTime"]) + offset,
  )

def toQuaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = {"w": 0, "x": 0, "y": 0, "z": 0}
    q["w"] = cr * cp * cy + sr * sp * sy
    q["x"] = sr * cp * cy - cr * sp * sy
    q["y"] = cr * sp * cy + sr * cp * sy
    q["z"] = cr * cp * sy - sr * sp * cy

    return q

def nan2None(obj):
    if isinstance(obj, dict):
        return {k:nan2None(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [nan2None(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

class Encoder(json.JSONEncoder):
    def encode(self, obj, *args, **kwargs):
        return super().encode(nan2None(obj), *args, **kwargs)
    # pylint: disable=method-hidden
    def default(self, o):
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return json.JSONEncoder.default(self, o)
