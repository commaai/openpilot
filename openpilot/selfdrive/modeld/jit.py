from tinygrad.engine.jit import CapturedJit, _TinyJit, _prepare_jit_inputs, graph_split_rewrite
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import Ops, UOp


def input_view(tensor: Tensor) -> Tensor:
  return Tensor(UOp.from_buffer(tensor._buffer())).reshape(tensor.shape)


def _ungraph(linear):
  calls = []
  for call in linear.src:
    if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == 'graph':
      calls.extend(_ungraph(call.src[0].src[0]).src)
    else:
      calls.append(call)
  return UOp(Ops.LINEAR, src=tuple(calls))


def _bind(jit, args, kwargs):
  inputs, values, names, info = _prepare_jit_inputs(args, kwargs)
  assert not values and names == jit.captured.expected_names and info == jit.captured.expected_input_info
  linear = _ungraph(jit.captured._linear)
  return linear.substitute({u: inputs[u.arg.slot] for u in linear.toposort(enter_calls=False) if u.op is Ops.PARAM}, walk=True)


def link_jits(*stages):
  # Bind fixed buffers once, then group the precompiled kernels into one execution graph.
  calls = tuple(call for jit, args, kwargs in stages for call in _bind(jit, args, kwargs).src)
  linear = UOp(Ops.LINEAR, src=calls)
  if any(call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == 'graph'
         for jit, _, _ in stages for call in jit.captured._linear.src):
    linear = graph_split_rewrite(linear)
  return _TinyJit(None, captured=CapturedJit(stages[-1][0].captured.ret, linear, [], []))
