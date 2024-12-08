import functools
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

class CUDALanguage(CStyleLanguage):
  kernel_prefix = "__global__ "
  smem_prefix = "__shared__ "
  smem_prefix_for_cast = False
  arg_int_prefix = "const int"
  barrier = "__syncthreads();" 
  float4 = "make_float4"
  gid = [f'blockIdx.{chr(120+i)}' for i in range(3)]
  lid = [f'threadIdx.{chr(120+i)}' for i in range(3)]
  xid = [f'(blockIdx.{chr(120+i)}*blockDim.{chr(120+i)}+threadIdx.{chr(120+i)})' for i in range(3)]
  half_prekernel = """
    #include <cuda_fp16.h>
    #include <mma.h>
    using namespace nvcuda;
    struct __align__(8) half4 {
      half2 x, y;
      __device__ __forceinline__ explicit half4(const float4& a): x(make_half2(__float2half(a.x), __float2half(a.y))), y(make_half2(__float2half(a.z),__float2half(a.w))) {}
      __device__ __forceinline__ explicit operator float4() const {return make_float4(__half2float(x.x), __half2float(x.y), __half2float(y.x), __half2float(y.y)); }
    };
    """ # if not getenv("PTX") else fromimport("tinygrad.renderer.assembly_ptx", "uops_to_ptx_asm") # assembly_ptx currently isn't supported

CUDARenderer = functools.partial(uops_to_cstyle, CUDALanguage())