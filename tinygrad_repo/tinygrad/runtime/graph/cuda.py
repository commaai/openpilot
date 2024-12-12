import ctypes
from typing import Any, Optional, Tuple, Dict, List, cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import init_c_var, dedup
from tinygrad.device import Buffer, Device
from tinygrad.runtime.ops_cuda import CUDADevice, check, encode_args, cu_time_execution
from tinygrad.ops import Variable
from tinygrad.engine.realize import ExecItem, BufferXfer, CompiledRunner
from tinygrad.engine.jit import MultiGraphRunner, GraphException

class CUDAGraph(MultiGraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)

    # Check all jit items are compatible.
    if not all(isinstance(ji.prg, (CompiledRunner, BufferXfer)) for ji in jit_cache): raise GraphException

    self.jc_idx_with_updatable_rawbufs = dedup([x[0] for x in self.input_replace.keys()])
    self.updatable_nodes: Dict[int, Tuple[Any, Any, Any, bool]] = {} # Dict[jc index] = tuple(graph node, node params, input kernel params, is memcpy)

    self.graph = init_c_var(cuda.CUgraph(), lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))

    for j,ji in enumerate(jit_cache):
      if isinstance(ji.prg, CompiledRunner):
        global_size, local_size = ji.prg.p.launch_dims(var_vals)

        new_node = cuda.CUgraphNode()
        deps = self._access_resources([x.base for x in ji.bufs if x is not None], ji.prg.p.outs, new_dependency=new_node)
        c_deps = (cuda.CUgraphNode*len(deps))(*deps) if deps else None

        c_args, vargs = encode_args([cast(Buffer, x)._buf for x in ji.bufs], [var_vals[x] for x in ji.prg.p.vars])
        kern_params = cuda.CUDA_KERNEL_NODE_PARAMS(ji.prg._prg.prg, *global_size, *local_size, 0, None, vargs)
        check(cuda.cuGraphAddKernelNode(ctypes.byref(new_node), self.graph, c_deps, len(deps), ctypes.byref(kern_params)))

        if j in self.launch_dims_replace or j in self.var_vals_replace or j in self.jc_idx_with_updatable_rawbufs:
          self.updatable_nodes[j] = (new_node, kern_params, c_args, False)
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        src_dev = cast(CUDADevice, Device[src.device])
        node_from = cuda.CUgraphNode()
        deps = self._access_resources(rawbufs=[dest.base, src.base], write=[0], new_dependency=node_from)
        c_deps = (cuda.CUgraphNode*len(deps))(*deps) if deps else None
        cp_params = cuda.CUDA_MEMCPY3D_v2(srcMemoryType=cuda.CU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                          dstMemoryType=cuda.CU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                          WidthInBytes=dest.nbytes, Height=1, Depth=1)
        check(cuda.cuGraphAddMemcpyNode(ctypes.byref(node_from), self.graph, c_deps, len(deps), ctypes.byref(cp_params), src_dev.context))
        if j in self.jc_idx_with_updatable_rawbufs: self.updatable_nodes[j] = (node_from, cp_params, src_dev.context, True)

    self.instance = init_c_var(cuda.CUgraphExec(), lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False) -> Optional[float]:
    # Update rawbuffers in the c_args struct.
    for (j,i),input_idx in self.input_replace.items():
      if not self.updatable_nodes[j][3]: setattr(self.updatable_nodes[j][2], f'f{i}', input_rawbuffers[input_idx]._buf)
      else:
        if i == 0: self.updatable_nodes[j][1].destDevice = input_rawbuffers[input_idx]._buf
        elif i == 1: self.updatable_nodes[j][1].srcDevice = input_rawbuffers[input_idx]._buf

    # Update var_vals in the c_args struct.
    for j, i, v in self.updated_vars(var_vals): setattr(self.updatable_nodes[j][2], f'v{i}', v)

    # Update launch dims in the kern_params struct.
    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      node = self.updatable_nodes[j][1]
      node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_dims, *global_dims # type: ignore[misc]

    # Update graph nodes with the updated structs.
    for node, c_node_params, c_args, is_copy in self.updatable_nodes.values():
      if not is_copy: check(cuda.cuGraphExecKernelNodeSetParams(self.instance, node, ctypes.byref(c_node_params)))
      else: check(cuda.cuGraphExecMemcpyNodeSetParams(self.instance, node, ctypes.byref(c_node_params), c_args))

    return cu_time_execution(lambda: check(cuda.cuGraphLaunch(self.instance, None)), enable=wait)

  def __del__(self):
    if hasattr(self, 'graph'): check(cuda.cuGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(cuda.cuGraphExecDestroy(self.instance))
