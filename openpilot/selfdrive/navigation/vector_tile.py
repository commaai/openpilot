"""Decode road polylines from Protomaps MVT tiles without a protobuf runtime."""
from dataclasses import dataclass

import numpy as np

from openpilot.selfdrive.navigation.pmtiles import read_varint


@dataclass
class Road:
  points: np.ndarray  # fractions of the tile, x right / y down
  kind: str
  detail: str
  tunnel: bool
  bridge: bool


def fields(data: bytes):
  offset = 0
  while offset < len(data):
    tag, offset = read_varint(data, offset)
    number, wire = tag >> 3, tag & 7
    if number == 0:
      raise ValueError("Invalid protobuf field")
    if wire == 0:
      value, offset = read_varint(data, offset)
    elif wire in (1, 2, 5):
      if wire == 2:
        length, offset = read_varint(data, offset)
      else:
        length = 8 if wire == 1 else 4
      if offset + length > len(data):
        raise ValueError("Truncated protobuf field")
      value = data[offset:offset + length]
      offset += length
    else:
      raise ValueError(f"Unsupported protobuf wire type: {wire}")
    yield number, wire, value


def integers(data: bytes) -> list[int]:
  result = []
  offset = 0
  while offset < len(data):
    value, offset = read_varint(data, offset)
    result.append(value)
  return result


def unzigzag(value: int) -> int:
  return (value >> 1) ^ -(value & 1)


def decode_value(data: bytes):
  """A string or boolean layer value. Other types are unused by the roads layer and decode to None."""
  for number, wire, value in fields(data):
    if number == 1 and wire == 2:
      return value.decode('utf-8')
    if number == 7 and wire == 0:
      return bool(value)
  return None


def decode_lines(commands: list[int]) -> list[np.ndarray]:
  lines = []
  line = []
  x = y = offset = 0
  while offset < len(commands):
    command = commands[offset]
    offset += 1
    kind, count = command & 7, command >> 3
    if kind not in (1, 2) or count == 0 or offset + 2 * count > len(commands):
      raise ValueError("Invalid MVT line geometry")
    if kind == 1 and count != 1:
      raise ValueError("LineString MoveTo must contain one point")
    if kind == 2 and not line:
      raise ValueError("LineTo before MoveTo")
    for _ in range(count):
      x += unzigzag(commands[offset])
      y += unzigzag(commands[offset + 1])
      offset += 2
      if kind == 1:
        if len(line) >= 2:
          lines.append(np.array(line, dtype=np.float64))
        line = []
      line.append((x, y))
  if len(line) >= 2:
    lines.append(np.array(line, dtype=np.float64))
  return lines


def decode_roads(data: bytes) -> list[Road]:
  roads = []
  for number, wire, layer in fields(data):
    if number != 3 or wire != 2:
      continue
    layer_fields = list(fields(layer))
    name = next((v.decode('utf-8') for n, w, v in layer_fields if n == 1 and w == 2), '')
    if name != 'roads':
      continue
    version = next((v for n, w, v in layer_fields if n == 15 and w == 0), 1)
    extent = next((v for n, w, v in layer_fields if n == 5 and w == 0), 4096)
    if version not in (1, 2) or extent <= 0:
      raise ValueError("Unsupported MVT roads layer")
    keys = [v.decode('utf-8') for n, w, v in layer_fields if n == 3 and w == 2]
    values = [decode_value(v) for n, w, v in layer_fields if n == 4 and w == 2]
    for n, w, feature in layer_fields:
      if n != 2 or w != 2:
        continue
      tags, geometry, geometry_type = [], [], 0
      for fn, fw, fv in fields(feature):
        if fn in (2, 4):
          target = tags if fn == 2 else geometry
          if fw == 2:
            target.extend(integers(fv))
          elif fw == 0:
            target.append(fv)
        elif fn == 3 and fw == 0:
          geometry_type = fv
      if geometry_type != 2:
        continue
      if len(tags) % 2:
        raise ValueError("Unpaired MVT tags")
      properties = {}
      for i in range(0, len(tags), 2):
        if tags[i] >= len(keys) or tags[i + 1] >= len(values):
          raise ValueError("MVT tag index out of bounds")
        properties[keys[tags[i]]] = values[tags[i + 1]]
      kind = properties.get('kind', '')
      if kind in ('highway', 'major_road', 'minor_road', 'other'):
        roads.extend(Road(points / extent, kind, properties.get('kind_detail', ''), properties.get('is_tunnel') is True,
                          properties.get('is_bridge') is True) for points in decode_lines(geometry))
  return roads
