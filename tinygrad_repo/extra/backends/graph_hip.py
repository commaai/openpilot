import ctypes
from typing import Tuple
import tinygrad.runtime.autogen.hip as hip
from tinygrad.helpers import init_c_var, time_execution_cuda_style
from tinygrad.runtime.ops_hip import check, hip_set_device
from tinygrad.runtime.graph.cuda import CUDAGraph

# TODO: this is only used in graph
def hip_time_execution(cb, enable=False): return time_execution_cuda_style(cb, hip.hipEvent_t, hip.hipEventCreate, hip.hipEventRecord, hip.hipEventSynchronize, hip.hipEventDestroy, hip.hipEventElapsedTime, enable=enable)  # noqa: E501

class HIPGraph(CUDAGraph):
  def __del__(self):
    if hasattr(self, 'graph'): check(hip.hipGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(hip.hipGraphExecDestroy(self.instance))
  def set_device(self): hip_set_device(self.dev)
  def encode_args_info(self): return (hip.hipDeviceptr_t, (1,2,3))
  def graph_create(self): return init_c_var(hip.hipGraph_t(), lambda x: check(hip.hipGraphCreate(ctypes.byref(x), 0)))
  def graph_instantiate(self, graph):
    return init_c_var(hip.hipGraphExec_t(), lambda x: check(hip.hipGraphInstantiate(ctypes.byref(x), graph, None, None, 0)))
  def graph_add_kernel_node(self, graph, c_deps, c_params):
    return init_c_var(hip.hipGraphNode_t(), lambda x: check(hip.hipGraphAddKernelNode(ctypes.byref(x), graph, c_deps, ctypes.sizeof(c_deps)//8 if c_deps else 0, ctypes.byref(c_params))))  # noqa: E501
  def graph_launch(self, *args, wait=False): return hip_time_execution(lambda: check(hip.hipGraphLaunch(*args)), enable=wait)
  def graph_exec_kernel_node_set_params(self, *args): return check(hip.hipGraphExecKernelNodeSetParams(*args))
  def build_kernel_node_params(self, prg, global_size, local_size, c_config):
    return hip.hipKernelNodeParams(hip.dim3(*local_size), c_config, ctypes.cast(prg.clprg.prg, ctypes.c_void_p), hip.dim3(*global_size), None, 0)
  def set_kernel_node_launch_dims(self, node, global_size: Tuple[int, int, int], local_size: Tuple[int, int, int]):
    node.blockDim.x, node.blockDim.y, node.blockDim.z, node.gridDim.x, node.gridDim.y, node.gridDim.z = *local_size, *global_size
