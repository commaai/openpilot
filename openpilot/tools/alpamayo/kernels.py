from functools import partial
from pathlib import Path

from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, Ops, UOp


def _cuda_source(*buffers, source, blocks, threads, name='alpamayo_gemv', outputs=1):
  inputs = buffers[outputs:]
  value = sum(x.flatten()[0].load().float() for x in inputs)
  sink = UOp.sink(*(out.flatten()[0].store(value) for out in buffers[:outputs]),
                  UOp.special(blocks, 'gidx0'), UOp.special(threads, 'lidx0'),
                  arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())), UOp(Ops.SOURCE, arg=source)))


def norm(x, weight, bias=None, eps=1e-6):
  width, rows = x.shape[-1], x.numel()//x.shape[-1]
  source = f'''
#include <cuda_bf16.h>
extern "C" __global__ __launch_bounds__(256) void alpamayo_norm(
    __nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* w{', const __nv_bfloat16* b' if bias is not None else ''}) {{
  int row=blockIdx.x*8+threadIdx.x/32, lane=threadIdx.x%32;
  if(row>={rows}) return;
  float values[{(width+31)//32}], mean=0, variance=0;
  #pragma unroll
  for(int j=0;j<{(width+31)//32};j++) {{
    int col=j*32+lane;
    values[j]=col<{width} ? float(x[row*{width}+col]) : 0;
    mean+=values[j];
  }}
  #pragma unroll
  for(int d=16;d;d/=2) mean+=__shfl_xor_sync(0xffffffff,mean,d);
  mean={'mean/'+str(width) if bias is not None else '0'};
  #pragma unroll
  for(int j=0;j<{(width+31)//32};j++) {{
    values[j]-=mean;
    if(j*32+lane<{width}) variance+=values[j]*values[j];
  }}
  #pragma unroll
  for(int d=16;d;d/=2) variance+=__shfl_xor_sync(0xffffffff,variance,d);
  float scale=rsqrtf(variance/{width}+{eps}f);
  #pragma unroll
  for(int j=0;j<{(width+31)//32};j++) {{
    int col=j*32+lane;
    float value=values[j]*scale;
    {'value=float(__float2bfloat16(value));' if bias is None else ''}
    if(col<{width}) out[row*{width}+col]=__float2bfloat16(value*float(w[col]){'+float(b[col])' if bias is not None else ''});
  }}
}}
'''
  out = Tensor.empty(x.shape, device=x.device, dtype=x.dtype)
  return Tensor.custom_kernel(out, x.contiguous(), weight, *(() if bias is None else (bias,)),
    fxn=partial(_cuda_source, source=source, blocks=(rows+7)//8, threads=256, name='alpamayo_norm'))[0]


def quantize(x):
  size = x.numel()
  blocks = (size+4095)//4096
  source = f'''
#include <cuda_bf16.h>
extern "C" __global__ __launch_bounds__(256) void alpamayo_amax(float* out, const __nv_bfloat16* x) {{
  __shared__ float warps[8];
  int tid=threadIdx.x;
  float maximum=0;
  #pragma unroll
  for(int j=0;j<16;j++) {{
    int i=blockIdx.x*4096+tid+j*256;
    if(i<{size}) maximum=fmaxf(maximum,fabsf(float(x[i])));
  }}
  #pragma unroll
  for(int d=16;d;d/=2) maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
  if(tid%32==0) warps[tid/32]=maximum;
  __syncthreads();
  if(tid<32) {{
    maximum=tid<8 ? warps[tid] : 0;
    #pragma unroll
    for(int d=16;d;d/=2) maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
    if(tid==0) out[blockIdx.x]=maximum;
  }}
}}
'''
  partials = Tensor.empty(blocks, device=x.device, dtype=dtypes.float)
  maximum = Tensor.custom_kernel(partials, x.contiguous(),
    fxn=partial(_cuda_source, source=source, blocks=blocks, threads=256, name='alpamayo_amax'))[0].max()
  scale = (maximum/448).clamp(min_=1e-12).contiguous()
  return (x.float()/scale).clamp(-448, 448).cast(dtypes.fp8e4m3).contiguous(), scale


def gemv(x, weight, scales=(), bias=None):
  rows, width = weight.shape
  dtype = '__nv_fp8_e4m3' if weight.dtype == dtypes.fp8e4m3 else '__nv_bfloat16'
  extra = (', const float* xs, const float* ws' if scales else '') + (', const __nv_bfloat16* bias' if bias is not None else '')
  source = f'''
#include <cuda_bf16.h>
#include <cuda_fp8.h>
extern "C" __global__ __launch_bounds__(128) void alpamayo_gemv(__nv_bfloat16* out, const {dtype}* x, const {dtype}* w{extra}) {{
  int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= {rows}) return;
  float acc = 0;
  for (int base = lane * 4; base < {width}; base += 128) {{
    #pragma unroll
    for (int j = 0; j < 4; j++) acc += float(x[base+j]) * float(w[row*{width}+base+j]);
  }}
  #pragma unroll
  for (int offset = 16; offset; offset /= 2) acc += __shfl_down_sync(0xffffffff, acc, offset);
  {'acc *= xs[0] * ws[0];' if scales else ''}
  {'acc += float(bias[row]);' if bias is not None else ''}
  if (lane == 0) out[row] = __float2bfloat16(acc);
}}
'''
  out = Tensor.empty(rows, device=x.device, dtype=dtypes.bfloat16)
  inputs = (x.contiguous(), weight, *scales) + (() if bias is None else (bias,))
  return Tensor.custom_kernel(out, *inputs, fxn=partial(_cuda_source, source=source, blocks=(rows+3)//4, threads=128))[0].reshape(*x.shape[:-1], rows)


def gemm(x, weight, scales, bias=None, heads=None):
  rows, width = x.shape
  columns = weight.shape[0]
  splits = 4 if rows <= 64 and columns <= 1024 and width >= 2048 else 1
  constants = {'M': rows, 'N': columns, 'K': width, 'BIAS': int(bias is not None), 'TRANSPOSE': int(heads is not None),
               'SEQUENCE': heads[1] if heads else 1, 'HEADS': heads[2] if heads else 1, 'HEAD_DIM': columns//heads[2] if heads else columns,
               'SPLITS': splits}
  source = '\n'.join(f'#define {name} {value}' for name, value in constants.items()) + '\n' + Path(__file__).with_name('gemm.cu').read_text()
  out = Tensor.empty(splits, rows, columns, device=x.device, dtype=dtypes.bfloat16 if splits == 1 else dtypes.float)
  out = Tensor.custom_kernel(out, x, weight, *scales, Tensor.zeros(1, device=x.device, dtype=dtypes.bfloat16) if bias is None else bias,
    fxn=partial(_cuda_source, source=source, blocks=splits*((rows+63)//64)*((columns+63)//64), threads=128, name='alpamayo_gemm'))[0]
  out = out.sum(0).cast(dtypes.bfloat16) if splits > 1 else out.reshape(rows, columns)
  return out.reshape(heads[0], heads[2], heads[1], columns//heads[2]).transpose(1, 2).reshape(rows, columns) if heads else out


def attention(q, k, v, mask=None, causal=False):
  from flash_attention import attention_program
  batch, heads, queries, width = q.shape
  kv_heads, keys = k.shape[1:3]
  splits = (keys+255)//256 if queries <= 64 else 1
  tile = 64 if queries > 64 else 16 if queries == 1 else 32
  shape = (batch, heads, kv_heads, queries, keys, width, splits, causal, mask is not None, tile)
  out = Tensor.empty(splits, *q.shape, device=q.device, dtype=q.dtype if splits == 1 else dtypes.float)
  stats = Tensor.empty(splits, batch, heads, queries, device=q.device, dtype=dtypes.float)
  mask = Tensor.zeros(1, device=q.device) if mask is None else mask.float().contiguous()
  out, stats = Tensor.custom_kernel(out, stats, q.contiguous(), k.contiguous(), v.contiguous(), mask,
    fxn=partial(attention_program, shape=shape, arch=int(Device[q.device].arch[3:]), blocks=splits*batch*heads*((queries+tile-1)//tile)))[:2]
  if splits == 1:
    return out.reshape(q.shape)
  size = batch*heads*queries*width
  source = f'''
#include <cuda_bf16.h>
extern "C" __global__ void alpamayo_attention_combine(__nv_bfloat16* out, const float* partial, const float* stats) {{
  int i = blockIdx.x*128+threadIdx.x;
  if (i >= {size}) return;
  int row = i/{width};
  float maximum = -__int_as_float(0x7f800000), total = 0, value = 0;
  #pragma unroll
  for (int s=0; s<{splits}; s++) maximum = fmaxf(maximum, stats[s*{size//width}+row]);
  #pragma unroll
  for (int s=0; s<{splits}; s++) {{
    float p = expf(stats[s*{size//width}+row]-maximum);
    total += p;
    value += p*partial[s*{size}+i];
  }}
  out[i] = __float2bfloat16(value/total);
}}
'''
  result = Tensor.empty(q.shape, device=q.device, dtype=q.dtype)
  return Tensor.custom_kernel(result, out, stats,
    fxn=partial(_cuda_source, source=source, blocks=(size+127)//128, threads=128, name='alpamayo_attention_combine'))[0]
