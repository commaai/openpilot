from typing import Callable, cast, Any
from tinygrad.dtype import AddrSpace, DType, PtrDType, ImageDType, dtypes, truncate
from tinygrad.helpers import DEBUG, OSX, unwrap, fromimport
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import GroupOp, Ops, UOp, PatternMatcher, UPat, range_str
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.support.c import POINTER
import base64, ctypes, ctypes.util, struct, functools, inspect, contextlib, itertools

def g(s:str): return getattr(mesa, s)
def nsrc(d:mesa.nir_def) -> mesa.nir_src: return mesa.nir_src(ssa=ctypes.pointer(d))

def glsl_type(t:DType): return mesa.glsl_array_type(glsl_type(t.base), t.size, 0).contents if isinstance(t, PtrDType) else {
  **{getattr(dtypes,k):g(f"glsl_type_builtin_{v}") for k,v in [('double','double'),('float','float'),('float16','float16_t'),('bool','uint8_t')]},
  **{d:g(f"glsl_type_builtin_{'u' * (d in dtypes.uints)}int{str(d.bitsize)+'_t' if d.itemsize != 4 else ''}") for d in dtypes.ints}}[t]

# alu ops, aop[<dtype>][<op>]
u_aop = { Ops.ADD: "iadd", Ops.MUL: "imul", Ops.IDIV: "udiv", Ops.MOD: "umod", Ops.CMPLT: "ult", Ops.CMPNE: "ine", Ops.CMPEQ: "ieq", Ops.OR: "ior",
          Ops.AND: "iand", Ops.XOR: "ixor", Ops.WHERE: "bcsel", Ops.MAX: "umax"}
s_aop = {**u_aop, Ops.CMPLT: "ilt", Ops.IDIV: "idiv", Ops.MOD: "irem", Ops.MAX: "imax"}
f_aop = { Ops.ADD: "fadd", Ops.MUL: "fmul", Ops.CMPLT: "flt", Ops.CMPNE: "fneu", Ops.CMPEQ: "feq", Ops.FDIV: "fdiv", Ops.RECIPROCAL: "frcp",
          Ops.MAX: "fmax", Ops.TRUNC: "ftrunc", Ops.SIN: "fsin", Ops.EXP2: "fexp2", Ops.LOG2: "flog2"}
aop = {**{x:u_aop for x in (dtypes.bool,)+dtypes.uints}, **{x:s_aop for x in dtypes.sints}, **{x:f_aop for x in dtypes.floats}}

def c(t:DType, u:bool=True) -> str: return "u" if t in dtypes.uints and u else ("i" if t in dtypes.ints else ("f" if t in dtypes.floats else "b"))
def ncast(b:mesa.nir_builder, src:mesa.nir_def, it:DType, ot:DType) -> mesa.nir_def:
  if isinstance(it, PtrDType) and ot == dtypes.long: return src
  if ot == dtypes.bool: return nalu(b, c(it, False)+'ne'+('u' if c(it) == 'f' else ''), src, nimm(b, 0, it))
  return nalu(b, f"{c(it)}2{c(it) if it in dtypes.ints and ot in dtypes.ints else c(ot, ot == dtypes.bool)}{ot.bitsize}", src)

def nif(b:mesa.nir_builder, cond:mesa.nir_def, then_fn:Callable, else_fn:Callable):
  nif = mesa.nir_push_if(b, cond)
  t = then_fn()
  mesa.nir_push_else(b, nif)
  e = else_fn()
  mesa.nir_pop_if(b, nif)
  return t, e

def nalu(b:mesa.nir_builder, op:str, *srcs:mesa.nir_def) -> mesa.nir_def: return g(f"nir_build_alu{len(srcs)}")(b, g(f"nir_op_{op}"), *srcs).contents

def nir_instr(nc=1, bs=lambda: None, intrins=None, srcs=None, has_def=True, df=None, also=lambda: None, **contents):
  def dec(f:Callable):
    @functools.wraps(f)
    def wrapper(*args, **kwargs) -> mesa.nir_def:
      (ba:=inspect.signature(f).bind(*args, **kwargs)).apply_defaults()
      def go(g): return g(**{nm: ba.arguments[nm] for nm in inspect.signature(g).parameters}) if callable(g) else g

      instr = f(*args, **kwargs)
      if has_def: mesa.nir_def_init(instr.contents.instr, instr.contents._def, go(nc), go(bs))
      for k, v in go(intrins or {}).items():
        idx = mesa.nir_intrinsic_infos[instr.contents.intrinsic].index_map[g(f"NIR_INTRINSIC_{k}")]
        assert idx > 0, "invalid intrinsic. mesa version mismatch?"
        instr.contents.const_index[idx - 1] = go(v)
      for i, src in enumerate(go(srcs or [])): ctypes.cast(instr.contents.src, ctypes.POINTER(mesa.nir_src))[i] = go(src)
      for k,v in {k:vcomp for k,v in contents.items() if (vcomp:=go(v)) is not None}.items(): setattr(instr.contents, k, go(v))
      mesa.nir_builder_instr_insert(ba.arguments['b'], instr.contents.instr)
      go(also)
      return instr.contents._def if has_def else (mesa.nir_def() if df is None else go(df))
    return wrapper
  return dec

@nir_instr(nc=1, bs=lambda src: src.bit_size, exact=lambda b:b.exact, fp_fast_math=lambda b:b.fp_fast_math)
def nchannel(b:mesa.nir_builder, src:mesa.nir_def, c:int):
  alu_src = mesa.nir_alu_src(src=nsrc(src))
  alu_src.swizzle[0] = c
  mov = mesa.nir_alu_instr_create(b.shader, mesa.nir_op_mov)
  ctypes.cast(mov.contents.src, ctypes.POINTER(mesa.nir_alu_src))[0] = alu_src
  return mov

def nimm_set(imm:mesa.nir_def, x, dtype:DType):
  instr = ctypes.cast(imm.parent_instr, ctypes.POINTER(mesa.nir_load_const_instr))
  struct.pack_into(unwrap(dtype.fmt), (ctypes.c_ubyte * dtype.itemsize).from_address(ctypes.addressof(instr.contents.value)), 0, truncate[dtype](x))

@nir_instr(nc=1, bs=lambda dtype: dtype.bitsize)
def nimm(b:mesa.nir_builder, x, dtype:DType) -> mesa.nir_def:
  nimm_set((instr:=mesa.nir_load_const_instr_create(b.shader, 1, dtype.bitsize)).contents._def, x, dtype)
  return instr
@nir_instr(nc=1, bs=lambda dtype: dtype.bitsize)
def nundef(b, dtype): return mesa.nir_undef_instr_create(b.shader, 1, dtype.bitsize)

deref_var = nir_instr(nc=1, bs=32, modes=lambda var:var.data.mode, type=lambda var:var.type, var=lambda var:ctypes.pointer(var))( # pylint: disable=W0108
  lambda b, var: mesa.nir_deref_instr_create(b.shader, mesa.nir_deref_type_var))

def iointr(space): return {"ALIGN_MUL":lambda dtype:dtype.itemsize} if space != AddrSpace.REG else {}
def scope(space): return 'global' if space == AddrSpace.GLOBAL else ('shared' if space == AddrSpace.LOCAL else 'deref')
nstore = nir_instr(has_def=False, df=lambda addr:addr, intrins=lambda space,val: {"WRITE_MASK":(1<<val.num_components)-1, **iointr(space)},
  num_components=lambda val:val.num_components, srcs=lambda space, addr, val: [nsrc(val), nsrc(addr)][::1 if space != AddrSpace.REG else -1])(
    lambda b, space, addr, val, dtype: mesa.nir_intrinsic_instr_create(b.shader, g(f"nir_intrinsic_store_{scope(space)}")))
nload = nir_instr(nc=lambda dtype:dtype.count, bs=lambda dtype:dtype.bitsize//dtype.count, num_components=lambda dtype:dtype.count,
  intrins=lambda space:{**({"ACCESS":mesa.ACCESS_CAN_REORDER} if space==AddrSpace.GLOBAL else {}), **iointr(space)}, srcs=lambda addr: [nsrc(addr)])(
    lambda b, space, addr, dtype: mesa.nir_intrinsic_instr_create(b.shader, g(f"nir_intrinsic_load_{scope(space)}")))

ngid = nir_instr(nc=3, bs=32)(lambda b: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_load_workgroup_id))
nlid = nir_instr(nc=3, bs=32)(lambda b: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_load_local_invocation_id))
ngsz = nir_instr(nc=3, bs=32)(lambda b: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_load_workgroup_size))
def nid(b): return nalu(b, "iadd", nalu(b, "imul", ngid(b), ngsz(b)), nlid(b))

nbarrier = nir_instr(has_def=False, intrins={"EXECUTION_SCOPE":mesa.SCOPE_WORKGROUP})(
  lambda b: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_barrier))

@nir_instr(has_def=False, target=lambda tgt:tgt and ctypes.pointer(tgt), condition=lambda cond:cond and nsrc(cond),
           else_target=lambda else_tgt: else_tgt and ctypes.pointer(else_tgt))
def njump(b:mesa.nir_builder, typ, tgt=None, cond=None, else_tgt=None): return mesa.nir_jump_instr_create(b.shader, typ)

def if_phi(b:mesa.nir_builder, cond, then_fn, else_fn): return mesa.nir_if_phi(b, *nif(b, cond, then_fn, else_fn)).contents

def nidx(b:mesa.nir_builder, buf, off, dtype, gate=None) -> mesa.nir_def:
  @nir_instr(nc=1, bs=32, modes=lambda buf: buf.data.mode, type=lambda buf: mesa.glsl_get_array_element(buf.type))
  def reg(b, buf):
    deref = mesa.nir_deref_instr_create(b.shader, mesa.nir_deref_type_array)
    deref.contents.parent, deref.contents.arr.index = nsrc(deref_var(b, buf)), nsrc(off)
    return deref
  f = (functools.partial(reg, b, buf) if dtype.addrspace == AddrSpace.REG else
       lambda: nalu(b, "iadd", buf, nalu(b, "imul", off, nimm(b, dtype.itemsize, dtypes.long))))
  return if_phi(b, gate, f, lambda: buf) if gate is not None else f()

class NIRRenderer(Renderer):
  suffix = "NIR"
  nir_options: bytes
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  code_for_op = {**{k:lambda:None for k in u_aop.keys()}, **{k:lambda:None for k in s_aop.keys()}, **{k:lambda:None for k in f_aop.keys()}}

  extra_matcher = PatternMatcher([
    # handle negative unsigned CONST
    (UPat.cvar("x", dtypes.uints), lambda x: UOp(Ops.CONST, dtype=x.dtype, arg=x.dtype.max+x.arg+1) if x.arg < 0 else None),
    # from ptx
    (UPat.var('x', dtype=dtypes.bool)<UPat.var('y'), lambda x,y: (x^True)&y),
    # load/store bool -> uint8
    (UPat(Ops.LOAD, dtypes.bool, name="x"),
     lambda x: x.replace(dtype=dtypes.uint8, src=x.src[0:1]+((x.src[1].cast(dtypes.uint8),) if len(x.src)>=2 else ())+x.src[2:]).cast(dtypes.bool)),
    (UPat(Ops.STORE, src=(UPat(), UPat(dtype=dtypes.bool)), name="x", allow_any_len=True),
     lambda x: x.replace(src=x.src[0:1] + (x.src[1].cast(dtypes.uint8),) + x.src[2:])),
    # load/store use pointer arithmetic, and the cast does nothing
    (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off")), allow_any_len=True, name="x"), lambda x,buf,off: x.replace(
      src=(buf,off.cast(dtypes.long))+x.src[2:]) if buf.dtype.addrspace != AddrSpace.REG and off.op not in (Ops.CAST, Ops.VECTORIZE) else None),
    (UPat(Ops.CAST, name="x"), lambda x: x.src[0] if isinstance(x.dtype, PtrDType) or x.src[0].dtype == dtypes.void else None),
  ])

  def_rewrite = PatternMatcher([
    (UPat(Ops.CONST, name="x"), lambda ctx,x: nimm(ctx.b, x.arg, x.dtype)),
    (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx.param(ctx.b, x, 8)),
    (UPat(Ops.DEFINE_VAR, name="x"), lambda ctx,x: ctx.param(ctx.b, x, 4)),
    (UPat(Ops.SPECIAL, name="x"), lambda ctx,x: nchannel(ctx.b, {'g':ngid, 'l':nlid, 'i': nid}[x.arg[0]](ctx.b), int(x.arg[-1]))),
    (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat.var("buf"),UPat.var("off")), allow_any_len=True), UPat.var("val")), allow_any_len=True, name="x"),
     lambda ctx,x,buf,off,val: nstore(ctx.b, buf.ptrdtype.addrspace, nidx(ctx.b, ctx.r[buf], ctx.r[off], buf.dtype), ctx.r[val], val.dtype)),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off"), UPat.var("gate"))), UPat.var("alt")), allow_any_len=True, name="x"),
     lambda ctx,x,buf,off,alt,gate: if_phi(ctx.b, ctx.r[gate],
      lambda: nload(ctx.b, buf.ptrdtype.addrspace, nidx(ctx.b, ctx.r[buf], ctx.r[off], buf.dtype, ctx.r[gate]), x.dtype), lambda: ctx.r[alt])),
    (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("off"))),), allow_any_len=True, name="x"),
     lambda ctx,x,buf,off: nload(ctx.b, buf.ptrdtype.addrspace, nidx(ctx.b, ctx.r[buf], ctx.r[off], buf.dtype), x.dtype)),
    (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: nalu(ctx.b, f"vec{x.dtype.count}", *[ctx.r[src] for src in x.src])),
    (UPat(GroupOp.ALU, name="x"), lambda ctx,x: nalu(ctx.b, aop[x.src[0].dtype.scalar()][x.op], *[ctx.r[src] for src in x.src])),
    (UPat(Ops.CAST, name="x"), lambda ctx,x: ncast(ctx.b, ctx.r[x.src[0]], x.src[0].dtype, x.dtype)),
    (UPat(Ops.BITCAST, src=(UPat.var("a"),), allow_any_len=True), lambda ctx,a: ctx.r[a]),
    (UPat(Ops.GEP, src=(UPat.var("a"),), name="x"), lambda ctx,x,a: nchannel(ctx.b, ctx.r[a], x.arg[0])),
    (UPat(Ops.DEFINE_REG, name="x"), lambda ctx,x:mesa.nir_local_variable_create(ctx.b.impl, glsl_type(x.dtype), f"acc{x.arg}".encode()).contents),
    (UPat(Ops.BARRIER), lambda ctx: nbarrier(ctx.b)),
    (UPat(Ops.IF, name="x"), lambda ctx,x: mesa.nir_push_if(ctx.b, ctx.r[x.src[0]])),
    (UPat(Ops.ENDIF, name="x"), lambda ctx,x: (lambda _: mesa.nir_def())(mesa.nir_pop_if(ctx.b, ctx.r[x.src[0]])))
  ])

  def __reduce__(self): return self.__class__, self.args

  def __init__(self, *args):
    self.compiler = fromimport("tinygrad.runtime.support.compiler_mesa", self.__class__.__name__.replace("Renderer", "Compiler"))(*args)
    self.args = args
    if hasattr(self.compiler, "nir_options"): self.nir_options = self.compiler.nir_options
    mesa.glsl_type_singleton_init_or_ref()

  def __del__(self):
    with contextlib.suppress(AttributeError): mesa.glsl_type_singleton_decref()

  def param(self, b:mesa.nir_builder, x, sz:int) -> mesa.nir_def: raise NotImplementedError("needs param")
  def prerender(self, uops:list[UOp]):
    self.b = mesa.nir_builder_init_simple_shader(mesa.MESA_SHADER_COMPUTE, mesa.nir_shader_compiler_options.from_buffer_copy(self.nir_options), None)
    self.b.shader.contents.info.workgroup_size_variable = any([u.op == Ops.SPECIAL and u.arg[0] == 'i' for u in uops])
  def postrender(self, uops:list[UOp]): pass

  def render(self, uops:list[UOp]):
    self.prerender(uops)
    for u in [u for u in uops if u.op is Ops.SPECIAL and u.arg[0] == "l"]: self.b.shader.contents.info.workgroup_size[int(u.arg[-1])] = u.src[0].arg
    self.r: dict[UOp, Any] = {}
    self.param_idx, ranges = 0, []

    for u in uops:
      if u.op in {Ops.NOOP, Ops.GROUP, Ops.INDEX}: pass
      elif u.op is Ops.AFTER:
        self.r[u] = self.r[u.src[0]]
      elif u.op == Ops.SINK:
        if u.arg is not None:
          self.b.shader.contents.info.name = ctypes.cast(ctypes.create_string_buffer(u.arg.function_name.encode()), POINTER[ctypes.c_char])
      elif u.op == Ops.DEFINE_LOCAL:
        self.r[u] = nimm(self.b, self.b.shader.contents.info.shared_size, dtypes.long)
        self.b.shader.contents.info.shared_size += u.dtype.nbytes()
      elif u.op == Ops.RANGE:
        ranges.append(i:=deref_var(self.b, mesa.nir_local_variable_create(self.b.impl, glsl_type(u.dtype), f"idx{range_str(u)}".encode()).contents))
        nstore(self.b, AddrSpace.REG, i, nimm(self.b, 0, u.dtype), u.dtype)
        mesa.nir_push_loop(self.b)
        self.r[u] = nload(self.b, AddrSpace.REG, i, u.dtype)
        nif(self.b, nalu(self.b, "ilt", self.r[u], self.r[u.src[0]]), lambda: None, lambda: njump(self.b, mesa.nir_jump_break))
      elif u.op == Ops.END:
        r = u.src[1]
        next_i = nalu(self.b, "iadd", self.r[r], nimm(self.b, 1, r.dtype))
        # TODO: this nif should be removable ... but TestMultiTensor.test_double_matmul_shard_W_0 segfaults with it gone
        nif(self.b, nalu(self.b, "ilt", next_i, self.r[r.src[0]]), lambda: None, lambda: njump(self.b, mesa.nir_jump_break))
        nstore(self.b, AddrSpace.REG, ranges.pop(), next_i, r.dtype),
        mesa.nir_pop_loop(self.b, None)
      else:
        if (d:=self.def_rewrite.rewrite(u, ctx=self)) is None: raise RuntimeError(f"failed to render {u.op} srcs {[x.dtype for x in u.src]}")
        self.r[u] = cast(mesa.nir_def, d)
    self.postrender(uops)

    mesa.nir_validate_shader(self.b.shader, b"after render")
    if DEBUG >= 4: mesa.nir_print_shader(self.b.shader, ctypes.POINTER(mesa.struct__IO_FILE).in_dll(ctypes.CDLL(ctypes.util.find_library('c')),
                                                                                                    "__stdoutp" if OSX else "stdout"))
    mesa.nir_serialize(blob:=mesa.struct_blob(), self.b.shader, False)
    ret = base64.b64encode(ctypes.string_at(blob.data, blob.size)).decode()

    mesa.ralloc_free(self.b.shader)
    ctypes.CDLL(None).free(blob.data)
    del self.b, self.r

    return ret

class NAKRenderer(NIRRenderer):
  device = "NV"

  param = nir_instr(nc=1, num_components=1, bs=lambda sz:sz*8, also=lambda self,sz: setattr(self, "param_idx", self.param_idx + sz),
    intrins={"ALIGN_MUL":lambda sz:sz}, srcs=lambda self,b: [nsrc(nimm(b, 0, dtypes.int)), nsrc(nimm(b, self.param_idx, dtypes.int))])(
       lambda self, b, x, sz: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_ldc_nv))

class LVPRenderer(NIRRenderer):
  device = "CPU"
  has_local = False
  has_shared = False
  global_max = (1, 0, 0)
  nir_options = mesa.lvp_nir_options
  # gallivm's exp2/log2 have "undefined behavior with infs, 0s and nans", so exp2(log2(0)*y) returns 0 instead of inf
  # https://gitlab.freedesktop.org/mesa/mesa/-/blob/c200b18e876468b51fe80d9660f612dc03a5138e/src/gallium/auxiliary/gallivm/lp_bld_arit.c#L2972
  code_for_op = {k:v for k,v in NIRRenderer.code_for_op.items() if k != Ops.EXP2}

  param = nir_instr(nc=1, bs=lambda sz: sz * 8, num_components=1, intrins={"ALIGN_MUL":lambda sz: sz, "RANGE":lambda self: self.param_sz},
    srcs=lambda b, self: [nsrc(nimm(b, 0, dtypes.int)), nsrc(nimm(b, self.param_idx, dtypes.int))], also=lambda self, sz:
    setattr(self, "param_idx", self.param_idx+sz))(lambda self,b,x,sz: mesa.nir_intrinsic_instr_create(b.shader, mesa.nir_intrinsic_load_ubo))

  def prerender(self, uops:list[UOp]):
    super().prerender(uops)
    self.param_sz = sum([8 if u.op == Ops.PARAM else u.dtype.itemsize for u in uops if u.op in (Ops.PARAM, Ops.DEFINE_VAR)])

# FIXME: this should be a rewrite rule
def tovec(b, coord): return nalu(b, "vec4", nchannel(b, coord, 0), nchannel(b, coord, 1), nundef(b, dtypes.int), nundef(b, dtypes.int))
def nfloat(dtype): return mesa.nir_type_float16 if dtype == dtypes.half else mesa.nir_type_float32
nstore_img = nir_instr(has_def=False, df=lambda img:img, num_components=lambda val:val.num_components,
  intrins=lambda dtype:{'IMAGE_DIM':mesa.GLSL_SAMPLER_DIM_2D, 'ACCESS':mesa.ACCESS_CAN_REORDER, 'SRC_TYPE':nfloat(dtype)},
  srcs=lambda b,img,coord,val:[nsrc(x) for x in [img, tovec(b, coord), nundef(b, dtypes.int), val, nimm(b, 0, dtypes.int)]])(
    lambda b,img,coord,val,dtype:mesa.nir_intrinsic_instr_create(b.shader,g("nir_intrinsic_image_store")))

_nload_img = nir_instr(intrins=lambda dtype:{'IMAGE_DIM':mesa.GLSL_SAMPLER_DIM_2D, 'ACCESS':mesa.ACCESS_CAN_REORDER, 'DEST_TYPE':nfloat(dtype)},
  nc=4, bs=32, num_components=4, srcs=lambda b,img,coord:[nsrc(x) for x in [img, tovec(b, coord), nundef(b, dtypes.int), nimm(b, 0, dtypes.int)]])(
    lambda b,img,coord,dtype: mesa.nir_intrinsic_instr_create(b.shader, g("nir_intrinsic_image_load")))

class IR3Renderer(NIRRenderer):
  device = "QCOM"
  has_aux = True

  def nload_img(ctx,img,coord):
    ctx.texs.add(img)
    return _nload_img(ctx.b, ctx.r[img], ctx.r[coord], img.dtype)

  def_rewrite = PatternMatcher([
    (UPat(Ops.STORE, src=(UPat.var('img').index(UPat.var('coord', dtypes.int.vec(2)), allow_any_len=True), UPat.var("val")),
          allow_any_len=True), lambda ctx,img,coord,val: nstore_img(ctx.b, ctx.r[img], ctx.r[coord], ctx.r[val], val.dtype)),
    (UPat(Ops.LOAD, src=(UPat.var('img').index(UPat.var('coord', dtypes.int.vec(2)), UPat.var("gate")), UPat.var("alt"))),
     lambda ctx,img,coord,alt,gate: if_phi(ctx.b, ctx.r[gate], lambda: ctx.nload_img(img, coord), lambda: ctx.r[alt])),
    (UPat(Ops.LOAD, src=(UPat.var('img').index(UPat.var('coord', dtypes.int.vec(2))),)), nload_img),
  ]) + NIRRenderer.def_rewrite

  _param = LVPRenderer.param
  def _param_img(self, x):
    self.img_idx += 1
    return nimm(self.b, self.img_idx - 1, dtypes.int)

  def param(self, b, x, sz): return self._param_img(x) if isinstance(x.dtype, ImageDType) else self._param(b, x, sz)

  def prerender(self, uops:list[UOp]):
    super().prerender(uops)
    self.texs:set[UOp] = set()
    self.uops, self.ibo_idx, self.img_idx = uops, 0, 0
    self.param_sz = sum([8 if u.op == Ops.PARAM else u.dtype.itemsize for u in uops if u.op in (Ops.PARAM, Ops.DEFINE_VAR)])

  def postrender(self, uops:list[UOp]):
    bufs, texs, imgs = [u for u in uops if u.op == Ops.PARAM], itertools.count().__next__, itertools.count().__next__
    for b in filter(lambda b: isinstance(b.dtype, ImageDType), bufs): nimm_set(self.r[b], texs() if b in self.texs else imgs(), dtypes.int)

    self.b.shader.contents.info.num_ubos = len([u for u in bufs if not isinstance(u.dtype, ImageDType)])
    self.b.shader.contents.info.num_images = texs() + imgs()

  def aux(self, uops:list[UOp]): return (tuple(u.dtype for u in uops if u.op == Ops.PARAM),)
