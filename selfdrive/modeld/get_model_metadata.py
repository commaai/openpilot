#!/usr/bin/env python3
import sys
import pathlib
import onnx
import codecs
import pickle

def get_name_and_shape(value_info:onnx.ValueInfoProto) -> tuple[str, tuple[int,...]]:
  shape = tuple([int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim])
  name = value_info.name
  return name, shape

if __name__ == "__main__":
  model_path = pathlib.Path(sys.argv[1])
  model = onnx.load(str(model_path))
  i = [x.key for x in model.metadata_props].index('output_slices')
  output_slices = model.metadata_props[i].value

  metadata = {}
  metadata['output_slices'] = pickle.loads(codecs.decode(output_slices.encode(), "base64"))
  metadata['input_shapes'] = dict([get_name_and_shape(x) for x in model.graph.input])
  metadata['output_shapes'] = dict([get_name_and_shape(x) for x in model.graph.output])

  metadata_path = model_path.parent / (model_path.stem + '_metadata.pkl')
  with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

  print(f'saved metadata to {metadata_path}')
