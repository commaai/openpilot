#!/usr/bin/env python3
"""Convert all float16 storage and type declarations in an ONNX graph to float32."""

import argparse
import pathlib

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper


def convert_tensor(tensor: TensorProto) -> bool:
  if tensor.data_type != TensorProto.FLOAT16:
    return False
  replacement = numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float32), name=tensor.name)
  tensor.CopyFrom(replacement)
  return True


def convert_type(type_proto) -> int:
  converted = 0
  if type_proto.HasField("tensor_type"):
    if type_proto.tensor_type.elem_type == TensorProto.FLOAT16:
      type_proto.tensor_type.elem_type = TensorProto.FLOAT
      converted += 1
  elif type_proto.HasField("sequence_type"):
    converted += convert_type(type_proto.sequence_type.elem_type)
  elif type_proto.HasField("optional_type"):
    converted += convert_type(type_proto.optional_type.elem_type)
  elif type_proto.HasField("map_type"):
    converted += convert_type(type_proto.map_type.value_type)
  return converted


def convert_attribute(attribute: AttributeProto) -> dict[str, int]:
  counts = {"tensors": 0, "types": 0, "casts": 0}
  if attribute.type == AttributeProto.TENSOR:
    counts["tensors"] += int(convert_tensor(attribute.t))
  elif attribute.type == AttributeProto.TENSORS:
    counts["tensors"] += sum(int(convert_tensor(tensor)) for tensor in attribute.tensors)
  elif attribute.type == AttributeProto.GRAPH:
    nested = convert_graph(attribute.g)
    for key in counts:
      counts[key] += nested[key]
  elif attribute.type == AttributeProto.GRAPHS:
    for graph in attribute.graphs:
      nested = convert_graph(graph)
      for key in counts:
        counts[key] += nested[key]
  elif attribute.type == AttributeProto.TYPE_PROTO:
    counts["types"] += convert_type(attribute.tp)
  elif attribute.type == AttributeProto.TYPE_PROTOS:
    counts["types"] += sum(convert_type(type_proto) for type_proto in attribute.type_protos)
  return counts


def convert_graph(graph) -> dict[str, int]:
  counts = {"tensors": 0, "types": 0, "casts": 0}
  for tensor in graph.initializer:
    counts["tensors"] += int(convert_tensor(tensor))
  for sparse in graph.sparse_initializer:
    counts["tensors"] += int(convert_tensor(sparse.values))

  for value_info in (*graph.input, *graph.output, *graph.value_info):
    counts["types"] += convert_type(value_info.type)

  for node in graph.node:
    for attribute in node.attribute:
      nested = convert_attribute(attribute)
      for key in counts:
        counts[key] += nested[key]
      if node.op_type == "Cast" and attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
        attribute.i = TensorProto.FLOAT
        counts["casts"] += 1
  return counts


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input", type=pathlib.Path)
  parser.add_argument("output", type=pathlib.Path)
  args = parser.parse_args()

  model = onnx.load(args.input, load_external_data=True)
  counts = convert_graph(model.graph)
  for function in model.functions:
    for node in function.node:
      for attribute in node.attribute:
        nested = convert_attribute(attribute)
        for key in counts:
          counts[key] += nested[key]
        if node.op_type == "Cast" and attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
          attribute.i = TensorProto.FLOAT
          counts["casts"] += 1

  onnx.checker.check_model(model, full_check=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  onnx.save(model, args.output)
  print(f"converted {counts['tensors']} tensors, {counts['types']} type declarations, and {counts['casts']} Cast nodes")
  print(f"saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
