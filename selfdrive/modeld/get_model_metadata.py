#!/usr/bin/env python3
import sys
import pathlib
import onnx
import codecs
import pickle
from typing import Any

def get_name_and_shape(value_info:onnx.ValueInfoProto) -> tuple[str, tuple[int,...]]:
  shape = tuple([int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim])
  name = value_info.name
  return name, shape

def get_metadata_value_by_name(model:onnx.ModelProto, name:str) -> str | Any:
  for prop in model.metadata_props:
    if prop.key == name:
      return prop.value
  return None

if __name__ == "__main__":
  model_path = pathlib.Path(sys.argv[1])
  model = onnx.load(str(model_path))
  output_slices = get_metadata_value_by_name(model, 'output_slices')
  assert output_slices is not None, 'output_slices not found in metadata'

  metadata = {
    'model_checkpoint': get_metadata_value_by_name(model, 'model_checkpoint'),
    'output_slices': pickle.loads(codecs.decode(output_slices.encode(), "base64")),
    'input_shapes': dict([get_name_and_shape(x) for x in model.graph.input]),
    'output_shapes': dict([get_name_and_shape(x) for x in model.graph.output])
  }

  metadata_path = model_path.parent / (model_path.stem + '_metadata.pkl')
  with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

  print(f'saved metadata to {metadata_path}')
