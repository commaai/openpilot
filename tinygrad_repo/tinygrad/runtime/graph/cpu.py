from typing import cast, TypeVar, Generic, get_args as get_typing_args
import itertools
from tinygrad.helpers import dedup, flatten, DEBUG, to_function_name
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.device import Buffer
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.ops import Variable
from tinygrad.dtype import DType, dtypes
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import LLVMRenderer, ldt

T = TypeVar('T')
class BatchedGraph(Generic[T], GraphRunner):
  def __init__(self, device, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[Variable, int]):
    renderer_class = get_typing_args(getattr(self, "__orig_bases__")[0])[0]
    if not issubclass(type(device.renderer), renderer_class) and not isinstance(device.renderer, renderer_class): raise GraphException

    super().__init__(jit_cache, input_rawbuffers, var_vals)
    self.base_bufs = dedup(b.base for ji in jit_cache for b in ji.bufs if b is not None and b not in input_rawbuffers)
    self.base_rawbufs = [b._buf for b in self.base_bufs]

    targs = [(f"arg{i}", x.dtype.ptr()) for i,x in enumerate(input_rawbuffers)] + \
            [(f"cbuf{i}", dtypes.char.ptr()) for i in range(len(self.base_bufs))] + \
            sorted([(f"{v.expr}", dtypes.int) for v in var_vals])
    code = self._prepare_code(device.renderer, jit_cache, input_rawbuffers, targs)
    if DEBUG >= 4: print(code)
    self.clprg = device.runtime("batched", device.compiler.compile_cached(code))

  def _prepare_code(self, renderer:T, jit_cache:list[ExecItem], input_rawbuffers:list[Buffer], targs:list[tuple[str, DType]]) -> str: return ""
  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False):
    return self.clprg(*[x._buf for x in rawbufs], *self.base_rawbufs, *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)], wait=wait)

class CPUGraph(BatchedGraph[ClangRenderer]):
  def _prepare_code(self, renderer:ClangRenderer, jit_cache:list[ExecItem], input_rawbuffers:list[Buffer], targs:list[tuple[str, DType]]) -> str:
    def render_arg(buf):
      if buf in input_rawbuffers: return f"arg{input_rawbuffers.index(buf)}"
      return f"({renderer.render_dtype(buf.dtype)}*)(cbuf{self.base_bufs.index(buf.base)} + {buf.offset})"

    batched = ["void batched("+','.join([f"{renderer.render_dtype(x[1])} {x[0]}" for x in targs])+") {"]
    for i, ji in enumerate(jit_cache):
      args = [render_arg(buf) for buf in ji.bufs] + [x.expr for x in cast(CompiledRunner, ji.prg).p.vars]
      batched.append(f"  {to_function_name(cast(CompiledRunner, ji.prg).p.name)}({','.join(args)});")
    batched.append("}")

    prep = [renderer._render(cast(CompiledRunner, ji.prg).p.uops or []) for i,ji in enumerate(jit_cache)]
    funcs = dedup(renderer._render_body(prep[i][0], *prep[i][1:], cast(CompiledRunner, ji.prg).p.uops,
                                        ["static", "__attribute__((always_inline))"]) for i,ji in enumerate(jit_cache))
    defines = dedup(itertools.chain.from_iterable(renderer._render_defines(cast(CompiledRunner, ji.prg).p.uops) for ji in jit_cache))
    entry = renderer._render_entry("batched", [(t[0], (t[1], False)) for t in targs])
    return '\n'.join(defines) + '\n' + '\n'.join([''.join(f) for f in funcs]) + '\n'.join(batched) + '\n' + entry

class LLVMGraph(BatchedGraph[LLVMRenderer]):
  def _prepare_code(self, renderer, jit_cache:list[ExecItem], input_rawbuffers:list[Buffer], targs:list[tuple[str, DType]]) -> str:
    out = []
    for i,ji in enumerate(jit_cache):
      args = []
      for j,buf in enumerate(cast(list[Buffer], ji.bufs)):
        arg = f"%arg{input_rawbuffers.index(buf)}" if buf in input_rawbuffers else f"%b{i}_{j}"
        if buf not in input_rawbuffers: out.append(f"  {arg} = getelementptr inbounds i8,ptr %cbuf{self.base_bufs.index(buf.base)},i64 {buf.offset}")
        args.append(f"{ldt(buf.dtype.ptr())} {arg}")
      args += [f"{ldt(x.dtype)} %{x.expr}" for x in cast(CompiledRunner, ji.prg).p.vars]
      out.append(f"  call void @{to_function_name(cast(CompiledRunner, ji.prg).p.name)}({','.join(args)})")

    kernels = dedup(tuple(renderer._render_kernel(cast(CompiledRunner, ji.prg).p.uops, ["internal"]) for i,ji in enumerate(jit_cache)))
    kernels += [((), renderer._render_fn("batched", [(f"%{x[0]}", x[1]) for x in targs], out))]
    assert flatten(x[0] for x in kernels) == [] # global definitions are not used in CPU mode right now
    return "\n".join([x[1] for x in kernels] + [renderer._render_footer(cast(CompiledRunner, ji.prg).p.uops)])
