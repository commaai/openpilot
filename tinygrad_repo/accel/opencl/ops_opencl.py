# type: ignore

from __future__ import annotations
import os
from tinygrad.llops.ops_gpu import GPUBuffer, CL, CLProgram, CLBuffer
from tinygrad.ops import ProcessingOps, ReduceOps, UnaryOps, BinaryOps
from tinygrad.helpers import prod, ConvArgs
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import pyopencl as cl

UNSAFE_FLOAT4 = int(os.getenv("UNSAFE_FLOAT4", 0))
NATIVE_EXPLOG = int(os.getenv("NATIVE_EXPLOG", 0))  # this is needed as a switch for the tests to pass
FLOAT16 = int(os.getenv("FLOAT16", 0))

import pathlib
def load(x):
   with open(x) as f:
     ret = f.read()
   return ret
CONV_SRC = load(pathlib.Path(__file__).resolve().parent.parent.parent / 'accel/opencl/conv.cl')
MATMUL_SRC = load(pathlib.Path(__file__).resolve().parent.parent.parent / 'accel/opencl/matmul.cl')

class CLImage:
  fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.HALF_FLOAT if FLOAT16 else cl.channel_type.FLOAT)

  def __init__(self, shape):
    self.max_hw = min(CL().cl_ctx.devices[0].image2d_max_width, CL.cl_ctx.devices[0].image2d_max_height)
    self.shape = shape
    self.n_tile = int(np.ceil(max(shape) / self.max_hw).item())
    # if n_tile > 1, we can't fit the image into a CL image at native size,
    # and need to internally store it as a set of disjoint tiles
    if self.n_tile * min(shape) > self.max_hw:
      raise Exception(f"shape {shape} exceeds Metal image limits, even after tiling")
    if shape[0] >= shape[1]:
      # wider than it is tall; extra tiles overflow on y
      self.tile_axis, tiled_width, tiled_height = 1, min(shape[0], self.max_hw), self.n_tile * shape[1]
    else:
      # taller than it is wide; extra tiles overflow on x
      self.tile_axis, tiled_width, tiled_height = 0, self.n_tile * shape[0], min(shape[1], self.max_hw)
    self.cl = cl.Image(CL.cl_ctx, cl.mem_flags.READ_WRITE, CLImage.fmt, shape=(tiled_width, tiled_height))
    CL.mem_used += self.cl.row_pitch * self.cl.height

  def pos_to_sample_pos(self, l="l", check_bounds=True):
    if self.n_tile == 1:
      # happy path where no indexing ops are needed
      return l
    # sad tiled path; need to adjust indices, and manually check bounds for the tiled axis
    if self.tile_axis == 1:
      sample_pos = f"((int2)({l}.x % {self.max_hw}, ({l}.x / {self.max_hw}) * {self.shape[1]} + {l}.y))"
      in_bounds = f"((0 <= {l}.y) && ({l}.y < {self.shape[1]}))"
    else:
      sample_pos = f"((int2)(({l}.y / {self.max_hw}) * {self.shape[0]} + {l}.x, {l}.y % {self.max_hw}))"
      in_bounds = f"((0 <= {l}.x) && ({l}.x < {self.shape[0]}))"
    if check_bounds:
      return f"({in_bounds} ? {sample_pos} : (int2)(-1, -1))"
    return sample_pos

  def __del__(self):
    if hasattr(self, "cl"):
      CL.mem_used -= self.cl.row_pitch * self.cl.height

def get_replacements(prg_src:str, opencl_type:List[str]) -> Dict[str, str]:
  middle_code = []

  """
  vv = "xyzw"
  for i in range(4):
    acc = f"outputValues[i].{vv[i%4]}"
    args = [x.split(" ")[-1].replace("*", "") for x in opencl_type]
    args = [f"(outputRow * get_image_width(output) + outputLocation.x)*4+{i}", acc]+args
    middle_code.append(f"{acc} = _ewop("+', '.join(args)+");\n")
  """
  acc = "outputValues[i]"
  args = [x.split(" ")[-1].replace("*", "") for x in opencl_type]
  args = ["smp", "outputLocation", "(outputLocation.y * get_image_width(output) + outputLocation.x)*4", acc]+args
  middle_code.append(f"{acc} = _ewop("+', '.join(args)+");\n")

  replacements = {}
  replacements["//PREFIX"] = prg_src
  replacements["//BINOP"] = ''.join(middle_code)
  if len(opencl_type) != 0:
    replacements["//ARGS"] = ","+','.join(opencl_type)
  return replacements

def get_getters(ewbufs, ret):
  fakebufs = []
  ewtypes = []
  getters = []
  for name, buf in ewbufs:
    view, unfolded, _ = buf.contiguous_view_constant_fold(name)
    if not unfolded:
      getters.append(view)
      fakebufs.append(name)
      getters.append(f"inline float4 get4_{name}(int gid) {{"+
        f"return (float4)(get_{name}(gid+0), get_{name}(gid+1), get_{name}(gid+2), get_{name}(gid+3)); }}")
    elif buf.is_image() and buf.shape == ret.shape and buf.st.contiguous:
      # use an image here
      ewtypes.append(f"read_only image2d_t {name}_g")
      getters.append(f"inline float4 get4_{name}(read_only image2d_t x, const sampler_t smp, int2 loc, int gid) {{ return read_imagef(x, smp, {buf._image.pos_to_sample_pos('loc')}); }}")
    elif buf.st.contiguous:
      # use float4
      ewtypes.append(f"__global const float4 *{name}_g")
      getters.append(f"inline float4 get4_{name}(__global const float4 *x, const sampler_t smp, int2 loc, int gid) {{ return x[gid/4]; }}")
    elif UNSAFE_FLOAT4:
      # aggressive constant folding
      fakebufs.append(name)
      prt = buf._backing.reshape((-1, 4))
      cc = []
      for ii in range(prt.shape[0]):
        cc.append("(float4)(%ff, %ff, %ff, %ff)" % (prt[ii][0], prt[ii][1], prt[ii][2], prt[ii][3]))
      getters.append(f"const __constant float4 const_{name}[] = {{"+', '.join(cc)+"};")
      getters.append(f"inline float4 get4_{name}(int gid) {{"+
        "int idx = gid;"+buf.st.expr()+";"+
        f"return const_{name}[idx/4]; }}")
      """
      # use float4 indexed (HACK!)
      # TODO: work out when this is okay
      ewtypes.append(f"__global const float4 *{name}_g")
      getters.append(f"inline float4 get4_{name}(__global const float4 *x, const sampler_t smp, int2 loc, int gid) {{"+
        "int valid = 1; int idx = gid;"+buf.st.expr()+";"+
        f"return x[idx/4]; }}")
      """
    else:
      # fallback to float
      getters.append(view)
      ewtypes.append(f"__global const float *{name}_g")
      getters.append(f"inline float4 get4_{name}(__global const float *x, const sampler_t smp, int2 loc, int gid) {{"+
        f"return (float4)(get_{name}(x,gid+0), get_{name}(x,gid+1), get_{name}(x,gid+2), get_{name}(x,gid+3)); }}")
  return fakebufs, ewtypes, getters

def roundup(x, n=4): return (x+(n-1))//n * n
class OpenCLBuffer(GPUBuffer):
  code_for_op = {
    UnaryOps.NOOP: "(A)", UnaryOps.NEG: "(-(A))", UnaryOps.RELU: "max(A, (float)0.)", UnaryOps.SIGN: "sign(A)",
    UnaryOps.EXP: "native_exp(A)" if NATIVE_EXPLOG else "exp(A)",
    UnaryOps.LOG: "native_log(A)" if NATIVE_EXPLOG else "log(A)",
    UnaryOps.RECIPROCAL: "native_recip(A)" if NATIVE_EXPLOG else "((float)1.0/A)",
    BinaryOps.ADD: "(A+B)", BinaryOps.SUB: "(A-B)", BinaryOps.MUL: "(A*B)", BinaryOps.DIV: "(A/B)", BinaryOps.POW: "pow(A,B)", BinaryOps.CMPEQ: "(A==B)",
    ReduceOps.SUM: "(acc + A)", ReduceOps.MAX: "max(A, acc)"
  }
  def __init__(self, shape, hostbuf:Optional[OpenCLBuffer]=None, backing:Optional[np.ndarray]=None):
    self._image = hostbuf._image if hostbuf is not None else None
    self.copied_backing = False
    super().__init__(shape, hostbuf, backing)
    assert not (self._image and self._buf)

  @staticmethod
  def fromCPU(x): return OpenCLBuffer(x.shape, backing=x.view(np.ndarray).astype(np.float32).ravel())
  
  def __repr__(self): return f"<OpenCLBuffer with shape {self.shape!r}>"

  @property
  def cl(self):
    if self._buf is None:
      if self._backing is not None and not self.copied_backing:
        self._buf = CLBuffer(4*roundup(prod(self._backing.shape)))
        CL.enqueue_copy(self._buf.cl, self._backing, is_blocking=False)
        self.copied_backing = True
      elif self.st.contiguous:
        self._buf = CLBuffer(4*roundup(prod(self.shape)))

      if self._image is not None:
        self._buf = CLBuffer(4*roundup(prod(self._image.shape)*4))
        if self._backing is not None and not self.copied_backing:
          CL.enqueue_copy(self._buf.cl, self._backing, is_blocking=False)
          self.copied_backing = True
        #print(f"converting {self.shape} back to buffer, image shape is {self._image.shape}")
        CLProgram("from_image", f"""
          __kernel void from_image(
              __global float4 *out,
              read_only image2d_t in) {{
            const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
            int2 l;
            l.y = get_global_id(1);
            l.x = get_global_id(0);
            int2 l_smp = {self._image.pos_to_sample_pos('l')};
            int W = {str(self._image.shape[0])};
            out[l.y*W + l.x] = read_imagef(in, smp, l_smp);
          }}
        """)(self._image.shape, None, self._buf.cl, self._image.cl)
        self._image = None
    return self._buf.cl
  
  def is_image(self): return self._image is not None

  @property
  def image(self):
    if self._image is None:
      assert len(self.shape) == 3 and self.shape[2] == 4, f"bad shape for image {self.shape}"
      assert self.st.contiguous, f"{self} is not contiguous"
      self._image = CLImage(shape=(self.shape[1], self.shape[0]))
      if self._buf is not None:
        assert prod(self.shape) <= prod(self._image.cl.shape)*4
        #print(f"converting {self.shape} to image with shape {self._image.shape}")
        CLProgram("to_image", f"""
          __kernel void to_image(
              write_only image2d_t out,
              __global const float4 *in) {{
            int2 l;
            l.y = get_global_id(1);
            l.x = get_global_id(0);
            int2 l_out = {self._image.pos_to_sample_pos('l', check_bounds=False)};
            int W = {str(self._image.shape[0])};
            write_imagef(out, l_out, in[l.y*W + l.x]);
          }}
        """)(self._image.shape, None, self._image.cl, self._buf.cl)
      self._buf = None
    return self._image.cl

  SUPPORTS_PADDING = True
  def processing_op(x, op:ProcessingOps, w:GPUBuffer, C:ConvArgs):
    assert op == ProcessingOps.CONV, f"{op} isn't supported"
    return type(x)(C.out_shape)._processing_op([("input", x.contiguous_op()), ("weight", w.contiguous_op())], "acc", C)

  def contiguous_view_constant_fold(x, name:str, reduce:Optional[int]=None) -> Tuple[str, Optional[str], str]:
    # this will only be for convs, for reduce we have to fall back to cl
    if x.is_image() and reduce is None:
      #print("is image")
      return f"""inline float get_{name}(const sampler_t smp, read_only image2d_t x, int gid) {{
        int valid = 1; int idx = gid; {x.st.expr().replace('//', '/')};
        int2 l;
        int W = {str(x._image.shape[0])};
        l.y = idx / (W*4);
        l.x = (idx/4) % W;
        int idx4 = idx % 4;
        int2 l_smp = {x._image.pos_to_sample_pos('l')};
        float4 dat = read_imagef(x, smp, l_smp);
        return valid ? (idx4 == 0 ? dat.x : (idx4 == 1 ? dat.y : (idx4 == 2 ? dat.z : dat.w))) : 0.0;
      }}""", f"read_only image2d_t {name}_g", f"get_{name}(smp, {name}_g, gid);"
    #ewtypes.append(f"read_only image2d_t {name}_g")
    return super().contiguous_view_constant_fold(name, reduce)

  def _processing_op(ret, bufs: List[Tuple[str, OpenCLBuffer]]=[], code:str="acc", C=None, op=ReduceOps.SUM, reduce_shape=None, earlybufs:Set[str]=set(), earlycode:str="acc"):
    if C is None or earlycode != "acc":
      # TODO: handle an opencl conv without the conv part
      return super()._processing_op(bufs, code, C, op, reduce_shape, earlybufs, earlycode)
    assert earlycode == "acc"

    x = [x for x in bufs if x[0] == "input"][0][1]
    w = [x for x in bufs if x[0] == "weight"][0][1]
    ewbufs = [x for x in bufs if x[0] not in ["input", "weight"]]

    # remove fakebufs
    fakebufs, ewtypes, getters = get_getters(ewbufs, ret)
    ewbufs = [x for x in ewbufs if x[0] not in fakebufs]

    elementwise_prefix = '\n'.join(getters)+ \
      "\n\ninline float4 _ewop("+','.join(["const sampler_t smp", "int2 loc", "int gid", "float4 acc"]+ewtypes)+") {\n"+ \
      ''.join([f"float4 {name} = get4_{name}(gid);\n" for name in fakebufs])+ \
      ''.join([f"float4 {name} = get4_{name}({name}_g, smp, loc, gid);\n" for name, _ in ewbufs])+ \
      f"return {code}; }}"

    replacements = get_replacements(elementwise_prefix, ewtypes)

    (x.image, w.image, ret.image)
    # fix sampling
    replacements["INPUT_LOCATION"] = x._image.pos_to_sample_pos("inputLocation")
    replacements["WEIGHT_LOCATION"] = w._image.pos_to_sample_pos("weightLocation")
    replacements["OUTPUT_LOCATION"] = ret._image.pos_to_sample_pos("outputLocation", check_bounds=False)
    # fix widths
    replacements["get_image_width(output)"] = f"({ret._image.shape[0]})"

    x, w = x.contiguous_op(), w.contiguous_op()
    options = []
    if C.bs > 1:
      options.append("-DBATCH")
      assert C.py == 0, "batched conv doesn't work with y-padding"
    if C.sx == 1 and C.sy == 1 and C.dx == 1 and C.dy == 1 and C.cin == 1:
      options.append("-DDEPTHWISE_UNSTRIDED")
    elif C.cin == 1:
      options.append("-DDEPTHWISE")
    if C.groups == 1 and C.H == 1 and C.W == 1 and C.iy == 1 and C.ix == 1 and C.oy == 1 and C.ox == 1 and C.sx == 1 and C.sy == 1 and C.dx == 1 and C.dy == 1 and C.bs == 1:
      options.append("-DMATMUL")
      # NOTE: this is not actually a matmul, it's a vector * matrix

      conv_args = []
      conv_short_names = ["numPackedInputChannelsForGroup", "totalNumPackedInputChannels", "numPackedOutputChannelsForGroup", "totalNumPackedOutputChannels", "numOutputColumns", "numOutputRows", "numInputRows"]
      conv_shorts = [max(1, C.cin//4), C.groups*C.cin//4, max(1, C.rcout//4), C.cout//4, C.ox, C.oy, C.iy]

      conv_src = MATMUL_SRC
      replacements["//SHORTS"] = ''.join([f"short {name} = {val};" for name,val in zip(conv_short_names, conv_shorts)])
      if "//BINOP" in replacements:
        replacements["//BINOP"] = replacements["//BINOP"].replace("outputValues[i]", "outputValues")
      for k,v in replacements.items():
        conv_src = conv_src.replace(k, v)

      #print(conv_src)
      conv_prg = CLProgram("matmul", conv_src,
        options=tuple(options),
        argdtypes=tuple([None, None, None, None] + [np.int16]*len(conv_args) + [None]*len(ewbufs))
      )
      global_work_size = [4, 16, C.cout//4]

      # must be even
      lw = CL.cl_ctx.devices[0].max_work_group_size // (global_work_size[0] * global_work_size[1])
      while global_work_size[2] % lw != 0:
        lw -= 1
      local_work_size = [4, global_work_size[1], lw]

      #print(global_work_size, local_work_size)
      conv_prg(global_work_size, local_work_size, ret.image, cl.LocalMemory(4 * local_work_size[0] * local_work_size[1] * lw), x.image, w.image, *conv_args, *[buf.image if 'image2d_t' in typ else buf.cl for typ, (_, buf) in zip(ewtypes, ewbufs)])
      return ret

    # this option is unused
    if C.H == 1 and C.W == 1:
      options.append("-DONLY_1X1_CONV")

    assert C.cout%4 == 0
    conv_src = CONV_SRC
    conv_short_names = ["filterSizeX", "filterSizeY", "paddingX", "paddingY", "strideX", "strideY", "dilationX", "dilationY"]
    conv_shorts = [C.W, C.H, C.px, C.py, C.sx, C.sy, C.dx, C.dy]
    conv_arg_names = ["numPackedInputChannelsForGroup", "totalNumPackedInputChannels", "numPackedOutputChannelsForGroup", "totalNumPackedOutputChannels", "numOutputColumns", "numOutputRows", "numInputRows"]
    conv_args = [max(1, C.cin//4), C.groups*C.cin//4, max(1, C.rcout//4), C.cout//4, C.ox, C.oy, C.iy]

    NUM_OUTPUTS = 4
    options.append(f"-DNUM_OUTPUTS={NUM_OUTPUTS}")

    # comment out for args
    conv_short_names += conv_arg_names
    conv_shorts += conv_args
    conv_args = []
    options.append("-DNOARGS")

    replacements["//SHORTS"] = ''.join([f"short {name} = {val};" for name,val in zip(conv_short_names, conv_shorts)])
    for k,v in replacements.items():
      conv_src = conv_src.replace(k, v)
    #print(conv_src)
    conv_prg = CLProgram("image_conv", conv_src,
      options=tuple(options),
      argdtypes=tuple([None, None, None] + [np.int16]*len(conv_args) + [None]*len(ewbufs))
    )
    global_work_size = [C.cout//4, (C.ox+NUM_OUTPUTS-1)//NUM_OUTPUTS, C.bs*C.oy]
    conv_prg(global_work_size, None, ret.image, x.image, w.image, *conv_args, *[buf.image if 'image2d_t' in typ else buf.cl for typ, (_, buf) in zip(ewtypes, ewbufs)])
    return ret

GPUBuffer = OpenCLBuffer
