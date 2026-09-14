import argparse
import codecs
import math
import pickle

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def add_history(model, frame_skip):
  specs = {v.name: (tuple(d.dim_value for d in v.type.tensor_type.shape.dim), v.type.tensor_type.elem_type)
           for v in model.graph.input}
  props = {p.key: p.value for p in model.metadata_props}
  slices = pickle.loads(codecs.decode(props['output_slices'].encode(), 'base64'))
  original_output = model.graph.output[0].name
  model = onnx.compose.add_prefix(model, 'model/', rename_functions=False)
  nodes, constants, inputs, outputs = [], [], [], []

  def op(kind, args, name=None, **attrs):
    name = name or f'queues/value_{len(nodes)}'
    nodes.append(helper.make_node(kind, args, [name], **attrs))
    return name

  def const(value):
    name = f'queues/constant_{len(constants)}'
    constants.append(numpy_helper.from_array(np.array(value, dtype=np.int64), name))
    return name

  def reshape(x, shape):
    return op('Reshape', [x, const(shape)])

  def slice_tensor(x, start, end, axis=0, step=1):
    return op('Slice', [x, const([start]), const([end]), const([axis]), const([step])])

  def input_tensor(name, shape, dtype=TensorProto.FLOAT):
    inputs.append(helper.make_tensor_value_info(name, dtype, shape))
    return name

  def queue(name, shape, value, dtype=TensorProto.FLOAT):
    state = input_tensor(f'state_{name}', shape, dtype)
    out = op('Concat', [slice_tensor(state, 1, shape[0]), value], f'next_{state}', axis=0)
    outputs.append(helper.make_tensor_value_info(out, dtype, shape))
    return out

  img, _ = specs['img']
  fb, _ = specs['features_buffer']
  dp, _ = specs['desire_pulse']
  feat_dim = math.prod(fb[2:])
  warped = input_tensor('warped', (2, 6, img[2], img[3]), TensorProto.UINT8)
  for i, name in enumerate(('img', 'big_img')):
    shape = (frame_skip * (img[1] // 6 - 1) + 1, 6, img[2], img[3])
    q = queue(f'{name}_q', shape, slice_tensor(warped, i, i + 1), TensorProto.UINT8)
    sampled = reshape(slice_tensor(q, 0, shape[0], step=frame_skip), img)
    op('Cast', [sampled], f'model/{name}', to=specs[name][1])

  desire = input_tensor('desire', (dp[2],))
  desire_q = queue('desire_q', (frame_skip * dp[1], dp[0], dp[2]), reshape(desire, (1, 1, -1)))
  pooled = op('ReduceMax', [reshape(desire_q, (dp[1], frame_skip, dp[0], dp[2])), const([1])], keepdims=0)
  op('Cast', [reshape(pooled, dp)], 'model/desire_pulse', to=specs['desire_pulse'][1])

  prev_feat = input_tensor('state_prev_feat', (fb[0], feat_dim))
  feat_q = queue('feat_q', (frame_skip * fb[1], fb[0], feat_dim), reshape(prev_feat, (1, fb[0], feat_dim)))
  sampled = reshape(slice_tensor(feat_q, 0, frame_skip * fb[1], step=frame_skip), fb)
  op('Cast', [sampled], 'model/features_buffer', to=specs['features_buffer'][1])
  for name in ('traffic_convention', 'action_t'):
    value = input_tensor(name, specs[name][0])
    op('Cast', [value], f'model/{name}', to=specs[name][1])

  nodes.extend(model.graph.node)
  op('Cast', [f'model/{original_output}'], 'outputs', to=TensorProto.FLOAT)
  hidden = slices['hidden_state']
  feature = reshape(slice_tensor('outputs', hidden.start, hidden.stop, axis=1), (fb[0], feat_dim))
  op('Identity', [feature], 'next_state_prev_feat')
  outputs.append(helper.make_tensor_value_info('next_state_prev_feat', TensorProto.FLOAT, (fb[0], feat_dim)))
  prediction_shape = tuple(d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim)
  outputs.insert(0, helper.make_tensor_value_info('outputs', TensorProto.FLOAT, prediction_shape))

  del model.graph.node[:]
  model.graph.node.extend(nodes)
  model.graph.initializer.extend(constants)
  del model.graph.input[:]
  model.graph.input.extend(inputs)
  del model.graph.output[:]
  model.graph.output.extend(outputs)
  onnx.checker.check_model(model)
  return model


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('input')
  p.add_argument('output')
  p.add_argument('--frame-skip', type=int, default=4)
  args = p.parse_args()
  model = add_history(onnx.load(args.input), args.frame_skip)
  onnx.save(model, args.output)
  print('inputs', [(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in model.graph.input])
  print('outputs', [v.name for v in model.graph.output])
