# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3

import os, pathlib, struct
from io import BufferedReader
from typing import Tuple, Union
from types import SimpleNamespace
from tinygrad.nn.state import TensorIO
from tinygrad.tensor import Tensor, dtypes

# Protobuf Wire Types
WIRETYPE_VARINT = 0; WIRETYPE_FIXED64 = 1; WIRETYPE_LENGTH_DELIMITED = 2; WIRETYPE_START_GROUP = 3; WIRETYPE_END_GROUP = 4; WIRETYPE_FIXED32 = 5 # noqa: E702

# TensorProto.DataType
class TensorDataType:
  UNDEFINED = 0; FLOAT = 1; UINT8 = 2; INT8 = 3; UINT16 = 4; INT16 = 5; INT32 = 6; INT64 = 7 # noqa: E702
  STRING = 8; BOOL = 9; FLOAT16 = 10; DOUBLE = 11; UINT32 = 12; UINT64 = 13; COMPLEX64 = 14; COMPLEX128 = 15; BFLOAT16 = 16 # noqa: E702

# AttributeProto.AttributeType
class AttributeType:
  UNDEFINED = 0; FLOAT = 1; INT = 2; STRING = 3; TENSOR = 4; GRAPH = 5; SPARSE_TENSOR = 11; TYPE_PROTO = 13; FLOATS = 6; INTS = 7 # noqa: E702
  STRINGS = 8; TENSORS = 9; GRAPHS = 10; SPARSE_TENSORS = 12; TYPE_PROTOS = 14 # noqa: E702

class PBType: FLOAT = 1; INT = 2; STRING = 3; FLOATS = 4; INTS = 5; STRINGS = 6; BYTES = 7; SUB = 8 # noqa: E702

PB_INFOS = {
  "OperatorSetIdProto": {1: ("domain", PBType.STRING), 2: ("version", PBType.INT)},
  "StringStringEntryProto": {1: ("key", PBType.STRING), 2: ("value", PBType.STRING)},
  # TODO: support uint64 parsing (11: "uint64_data") and double parsing (10: "double_data")
  "TensorProto": {1: ("dims", PBType.INT, True), 2: ("data_type", PBType.INT), 4: ("float_data", PBType.FLOATS),
    13: ("external_data", PBType.SUB, True, "StringStringEntryProto"), 14: ("data_location", PBType.INT),
    5: ("int32_data", PBType.INTS), 7: ("int64_data", PBType.INTS), 8: ("name", PBType.STRING), 9: ("raw_data", PBType.BYTES)},
  "TensorShapeProtoDimension": {1: ("dim_value", PBType.INT), 2: ("dim_param", PBType.STRING)},
  "TensorShapeProto": {1: ("dim", PBType.SUB, True, "TensorShapeProtoDimension")},
  "ModelProto": {1: ("ir_version", PBType.INT), 5: ("model_version", PBType.INT),
    2: ("producer_name", PBType.STRING), 3: ("producer_version", PBType.STRING), 4: ("domain", PBType.STRING), 6: ("doc_string", PBType.STRING),
    7: ("graph", PBType.SUB, False, ("GraphProto", lambda: {"node": [], "initializer": [], "input": [], "output": [], "value_info": []})),
    8: ("opset_import",PBType.SUB, True, "OperatorSetIdProto")},
  "GraphProto": {2: ("name", PBType.STRING), 10: ("doc_string", PBType.STRING),
    1: ("node", PBType.SUB, True, ("NodeProto", lambda: {"input": [], "output": [], "attribute": [], "domain": None})),
    5: ("initializer", PBType.SUB, True, ("TensorProto", lambda: {"dims": [], "float_data": [], "int32_data": [], "string_data": [],
                                                                  "int64_data": [], "double_data": [], "uint64_data": []})),
    11: ("input", PBType.SUB, True, "ValueInfoProto"), 12: ("output", PBType.SUB, True, "ValueInfoProto")},
  "NodeProto": { 1: ("input", PBType.STRING, True), 2: ("output", PBType.STRING, True), 3: ("name", PBType.STRING),
    4: ("op_type", PBType.STRING), 6: ("doc_string", PBType.STRING), 7: ("domain", PBType.STRING),
    5: ("attribute", PBType.SUB, True, ("AttributeProto", lambda: {"floats": [], "ints": [], "strings": []}))},
  "AttributeProto": {1: ("name", PBType.STRING), 20: ("type", PBType.INT), 3: ("i", PBType.INT), 8: ("ints", PBType.INT, True),
    2: ("f", PBType.FLOAT), 7: ("floats", PBType.FLOAT, True), 4: ("s", PBType.BYTES), 9: ("strings", PBType.BYTES, True),
    5:("t", PBType.SUB, False, ("TensorProto", lambda: {"dims": [], "float_data": [], "int32_data": [], "string_data": [], "int64_data": [],
                                                        "double_data": [], "uint64_data": []}))},
  "ValueInfoProto": {1: ("name", PBType.STRING), 2: ("type", PBType.SUB, False, "TypeProto"), 3: ("doc_string", PBType.STRING)},
  "TypeProto": {1: ("tensor_type", PBType.SUB, False, "TypeProtoTensor"), 4: ("sequence_type", PBType.SUB, False, "TypeProtoSequence"),
    9: ("optional_type", PBType.SUB, False, "TypeProtoOptional"), 6: ("denotation", PBType.STRING)},
  "TypeProtoSequence": {1: ("elem_type", PBType.SUB, False, "TypeProto")},
  "TypeProtoOptional": {1: ("elem_type", PBType.SUB, False, "TypeProto")},
  "TypeProtoTensor": {1: ("elem_type", PBType.INT), 2: ("shape", PBType.SUB, False, ("TensorShapeProto", lambda: {"dim": []}))},
}

def onnx_load(fn: Union[Tensor, str, pathlib.Path], load_external_data: bool=True):
  parser = OnnxParser(fn, load_external_data)
  onnx_model = parser.parse()
  model = dict_to_namespace(onnx_model)
  return model

def gen_result(obj: dict, key_name, val, repeated: bool):
  if repeated: obj.setdefault(key_name, []).append(val)
  else: obj[key_name] = val

def dict_to_namespace(d):
  if isinstance(d, dict): return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
  elif isinstance(d, list): return [dict_to_namespace(i) for i in d]
  return d

class OnnxParser:
  def __init__(self, inp: Union[Tensor, str, pathlib.Path], load_external_data: bool=True):
    self.file_path: Union[pathlib.Path, None] = None
    self.load_external_data = load_external_data
    if not isinstance(inp, Tensor):
      self.file_path = pathlib.Path(inp)
      self.tensor = Tensor(self.file_path)
    else: self.tensor = inp
    self.attr_func_dict = { PBType.BYTES: self._handle_bytes, PBType.SUB: self._handle_sub_message, PBType.FLOATS: self._handle_packed_floats,
      PBType.INT: self._handle_int64, PBType.INTS: self._handle_packed_int64s, PBType.STRING: self._handle_string, PBType.FLOAT: self._handle_float}
    self.registered_handles = {}
    for pb_name in PB_INFOS:
      res = {}
      for fid, config in PB_INFOS[pb_name].items():
        parser_fn, repeated = None, False
        if len(config) == 2: name, attr = config
        elif len(config) == 3: name, attr, repeated = config
        elif len(config) == 4: name, attr, repeated, parser_fn = config
        handler_fn = self.attr_func_dict[attr]
        def _wrapper_handler(obj, reader, wt, h=handler_fn, n=name, p=parser_fn, r=repeated): return h(obj, n, reader, wt, parser_func=p, repeated=r)
        _wrapper_handler._debug_info = f"{fid}, {name} => {handler_fn}"
        res[fid] = _wrapper_handler
      self.registered_handles[pb_name] = res

  def parse(self):
    reader = BufferedReader(TensorIO(self.tensor))
    return self._parse_message(reader, "ModelProto", lambda: {"opset_import": [], "domain": None, "graph": None})

  def decode_varint(self, reader: BufferedReader) -> int:
    result = 0
    shift = 0
    while True:
      data = reader.read(1)
      if data == b"": raise EOFError("decode_varint EOF")
      result |= (data[0] & 0x7F) << shift
      if not (data[0] & 0x80): return result
      shift += 7
      if shift >= 70: raise ValueError("Varint too long")

  def skip_field_value(self, reader: BufferedReader, wire_type):
    if wire_type == WIRETYPE_VARINT: self.decode_varint(reader)
    elif wire_type == WIRETYPE_FIXED64: reader.seek(8, os.SEEK_CUR)
    elif wire_type == WIRETYPE_FIXED32: reader.seek(4, os.SEEK_CUR)
    elif wire_type == WIRETYPE_LENGTH_DELIMITED: reader.seek(self.decode_varint(reader), os.SEEK_CUR)
    else: raise ValueError(f"Unknown wire type: {wire_type}")

  def _parse_message(self, reader, message_field_handlers_name, initial_obj_factory=lambda: {}):
    message_field_handlers = self.registered_handles[message_field_handlers_name]
    obj = initial_obj_factory()
    while True:
      try:
        tag_val = self.decode_varint(reader)
        field_number = tag_val >> 3
        wire_type = tag_val & 0x07
        if handler := message_field_handlers.get(field_number):
          handler(obj, reader, wire_type)
        else: self.skip_field_value(reader, wire_type)
      except EOFError: break
    if message_field_handlers_name == "TensorProto" and self.load_external_data and obj.get("data_location", 0) == 1: self._parse_external_data(obj)
    return obj

  def _handle_delimited(self, reader:BufferedReader, use_tensor=False) -> Tuple[bytes, Tensor]:
    str_len = self.decode_varint(reader)
    if not use_tensor: return reader.read(str_len)
    res = reader.raw._tensor[reader.tell():(reader.tell()+str_len)]
    reader.seek(str_len, os.SEEK_CUR)
    return res

  def _handle_string(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for string field '{key_name}'")
    value = self._handle_delimited(reader)
    gen_result(obj, key_name, value.decode("utf-8"), repeated)

  def _handle_bytes(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for bytes field '{key_name}'")
    value = self._handle_delimited(reader, use_tensor=True)
    gen_result(obj, key_name, value, repeated)

  def _handle_int64(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_VARINT: raise ValueError(f"Expected varint for int64 field '{key_name}'")
    val = self.decode_varint(reader)
    gen_result(obj, key_name, val - 2**64 if val & (1 << 63) else val, repeated)

  def _handle_float(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_FIXED32: raise ValueError(f"Expected fixed32 for float field '{key_name}'")
    val, = struct.unpack("<f", reader.read(4))
    gen_result(obj, key_name, val, repeated)

  def _handle_packed_int64s(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed int64s expected length_delimited")
    total_bytes_len = self.decode_varint(reader)
    old_pos = reader.tell()
    values = []
    while reader.tell() < total_bytes_len + old_pos:
      val = self.decode_varint(reader) # need copy here because packed ints are varint
      values.append(val - 2**64 if val & (1 << 63) else val)
    obj[key_name] = Tensor(values, dtype=dtypes.int64)

  def _handle_packed_floats(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError("Packed floats expected length_delimited")
    value = self._handle_delimited(reader, use_tensor=True)
    obj[key_name] = value.bitcast(dtypes.float32)

  def _handle_sub_message(self, obj, key_name, reader, wire_type, parser_func=None, repeated=False):
    if wire_type != WIRETYPE_LENGTH_DELIMITED: raise ValueError(f"Expected length-delimited for sub-message field '{key_name}'")
    value = self._handle_delimited(reader, use_tensor=True)
    if isinstance(parser_func, str): sub_obj = self._parse_message(BufferedReader(TensorIO(value)), parser_func)
    elif isinstance(parser_func, tuple): sub_obj = self._parse_message(BufferedReader(TensorIO(value)), parser_func[0], parser_func[1])
    else: sub_obj = parser_func(BufferedReader(TensorIO(value)))
    gen_result(obj, key_name, sub_obj, repeated)

  def _parse_external_data(self, obj):
    if "external_data" not in obj: raise ValueError("no external_data")
    location = None
    length = None
    offset = 0
    for kv in obj["external_data"]:
      if kv["key"] == "location": location = kv["value"]
      if kv["key"] == "offset": offset = int(kv["value"])
      if kv["key"] == "length": length = int(kv["value"])
    if location is None: raise ValueError("no location in external_data")
    if self.file_path is None:
      # get onnx file path from Tensor
      if isinstance(self.tensor.device, str) and self.tensor.device.startswith("DISK:"):
        self.file_path = self.tensor.device[5:]
        if not (ext_path := self.file_path.parent.joinpath(location)).exists():
          raise Exception(f"external location not exists: {ext_path}, may caused by symbolic link, try passing onnx file path to onnx_load")
      else: raise Exception("onnx external_data need the origin file path, try passing onnx file path to onnx_load")
    ext_path = self.file_path.parent.joinpath(location)
    if not ext_path.exists(): raise Exception(f"external location not exists: {ext_path}")
    ext_tensor = Tensor(ext_path)
    obj["raw_data"] = ext_tensor[offset:offset+length] if length is not None else ext_tensor[offset:]
    obj["data_location"] = 0
