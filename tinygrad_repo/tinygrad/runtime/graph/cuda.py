import ctypes
from typing import Any, cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.runtime.support.c import init_c_var
from tinygrad.device import Device, MultiBuffer
from tinygrad.uop.ops import UOp, Ops
from tinygrad.runtime.ops_cuda import CUDADevice, check, encode_args, cu_time_execution
from tinygrad.engine.jit import MultiGraphRunner

class CUDAGraph(MultiGraphRunner):
  def __init__(self, linear, input_uops=()):
    super().__init__(linear, input_uops)

    self.nodes: list[tuple[Any, ...]] = [] # list of tuple(graph node, node params, c_args/context, is memcpy)
    self.graph = init_c_var(cuda.CUgraph, lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))

    for (dev_idx, ast, bufs, device_vars), runtime in zip(self.calls, self.runtimes):
      if ast.op is Ops.PROGRAM:
        assert runtime is not None
        global_size, local_size = ast.arg.launch_dims({v: 0 for v in self.vars})

        c_deps, new_node = self.new_node([b.base for b in bufs], ast.arg.outs)
        c_args, vargs = encode_args([b._buf for b in bufs], [device_vars.get(x.expr, 0) for x in ast.arg.vars])
        kern_params = cuda.CUDA_KERNEL_NODE_PARAMS_v1(runtime.prg, *global_size, *local_size, 0,
                                                      ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p)), vargs)
        check(cuda.cuGraphAddKernelNode(ctypes.byref(new_node), self.graph, c_deps, len(c_deps or []), ctypes.byref(kern_params)))

        self.nodes.append((new_node, kern_params, c_args, False))
      elif ast.op is Ops.COPY:
        dest, src = bufs[0], bufs[1]
        src_dev = cast(CUDADevice, Device[src.device])
        c_deps, new_node = self.new_node([dest.base, src.base], [0])
        cp_params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                          dstMemoryType=cuda.CU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                          WidthInBytes=dest.nbytes, Height=1, Depth=1)
        check(cuda.cuGraphAddMemcpyNode(ctypes.byref(new_node), self.graph, c_deps, len(c_deps or []), ctypes.byref(cp_params), src_dev.context))

        self.nodes.append((new_node, cp_params, src_dev.context, True))

    self.instance = init_c_var(cuda.CUgraphExec, lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))
    self.updatable = sorted({j for j,r in enumerate(self.uop_replace) if r} | self.var_vals_replace.keys() | self.launch_dims_replace.keys())

  def new_node(self, bufs, write):
    deps = self._access_resources(bufs, write, new_dependency=(node:=cuda.CUgraphNode()))
    return (cuda.CUgraphNode*len(deps))(*deps) if deps else None, node

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False):
    # Update buffers in the c_args struct.
    for j in self.updatable:
      (_, params, c_args, is_copy), dev_idx = self.nodes[j], self.calls[j][0]
      for pos, iidx in self.uop_replace[j]:
        buf = b.bufs[dev_idx] if isinstance(b:=input_uops[iidx].buffer, MultiBuffer) else b
        if not is_copy: setattr(c_args, f'f{pos}', buf._buf)
        else: setattr(params, 'srcDevice' if pos == 1 else 'dstDevice', buf._buf)

    # Update var_vals in the c_args struct.
    for j, i, v in self.updated_vars(var_vals): setattr(self.nodes[j][2], f'v{i}', v)

    # Update launch dims in the kern_params struct.
    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      node = self.nodes[j][1]
      node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_dims, *global_dims # type: ignore[misc]

    # Update graph nodes with the updated structs.
    for j in self.updatable:
      node, c_node_params, c_args, is_copy = self.nodes[j]
      if not is_copy: check(cuda.cuGraphExecKernelNodeSetParams(self.instance, node, ctypes.byref(c_node_params)))
      else: check(cuda.cuGraphExecMemcpyNodeSetParams(self.instance, node, ctypes.byref(c_node_params), c_args))

    return cu_time_execution(lambda: check(cuda.cuGraphLaunch(self.instance, None)), enable=wait)

  def __del__(self):
    if hasattr(self, 'graph'): check(cuda.cuGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(cuda.cuGraphExecDestroy(self.instance))
