import sys, onnx, time
from tinygrad import Tensor, TinyJit, Device, GlobalCounters, fetch
from tinygrad.tensor import _from_np_dtype
from extra.onnx import get_run_onnx

if __name__ == "__main__":
  onnx_file = fetch(sys.argv[1])
  print(onnx_file)
  onnx_model = onnx.load(onnx_file)
  Tensor.no_grad = True
  Tensor.training = False
  run_onnx = get_run_onnx(onnx_model)
  print("loaded model")

  # find preinitted tensors and ignore them
  initted_tensors = {inp.name:None for inp in onnx_model.graph.initializer}
  expected_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initted_tensors]

  # get real inputs
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in expected_inputs}
  input_types = {inp.name:onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in expected_inputs}
  run_onnx_jit = TinyJit(lambda **kwargs: next(iter(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values())), prune=True)

  for i in range(3):
    new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in sorted(input_shapes.items())}
    GlobalCounters.reset()
    print(f"run {i}")
    run_onnx_jit(**new_inputs)

  # run 20 times
  for _ in range(20):
    new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in sorted(input_shapes.items())}
    GlobalCounters.reset()
    st = time.perf_counter()
    out = run_onnx_jit(**new_inputs)
    mt = time.perf_counter()
    val = out.numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")
