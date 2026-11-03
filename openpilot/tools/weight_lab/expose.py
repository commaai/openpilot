#!/usr/bin/env python3
"""Write a copy of the ONNX model that also outputs chosen hidden tensors (as x_<name>).

A tap is either the input of the layer that holds a weight, or a tensor name given as 'tensor:<name>'.
"""
import sys
import onnx

TAPS = {
  'plan_feat': 'policy_model.temporal_hydra.in_layer.plan.weight',      # 512 temporal feature into plan head
  'plan_trunk': 'policy_model.temporal_hydra.final_layer.plan.weight',  # 256 plan trunk into final layer
  'plan_trunk_pre': 'tensor:model/p_add_7',                             # 256 plan trunk before its last ReLU
  'lead_feat': 'vision_model.policy.hydra.in_layer.lead.weight',        # 512 per-frame vision feature into lead head
  'lead_trunk': 'vision_model.policy.hydra.final_layer.lead.weight',    # 64 lead trunk into final layer
}


def main(src, dst):
  m = onnx.load(src)
  g = m.graph
  for tap, wname in TAPS.items():
    if wname.startswith('tensor:'):
      x = wname.split(':', 1)[1]
    else:
      x = next(n for n in g.node if any(i.endswith(wname) for i in n.input)).input[0]
    g.node.append(onnx.helper.make_node('Cast', [x], [f'x_{tap}'], name=f'tap_{tap}', to=onnx.TensorProto.FLOAT))
    g.output.append(onnx.helper.make_tensor_value_info(f'x_{tap}', onnx.TensorProto.FLOAT, None))
    print(tap, '<-', x)
  onnx.save(m, dst)


if __name__ == '__main__':
  main(*sys.argv[1:3])
