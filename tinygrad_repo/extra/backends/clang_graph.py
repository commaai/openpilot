from typing import List, Dict, cast
import ctypes
from tinygrad.helpers import dedup, cpu_time_execution, DEBUG
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.device import Buffer, Device
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.ops import Variable
from tinygrad.runtime.ops_cpu import ClangProgram
from tinygrad.renderer.cstyle import ClangRenderer
render_dtype = ClangRenderer().render_dtype

class ClangGraph(GraphRunner):
  def __init__(self, jit_cache: List[ExecItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    if not all(isinstance(ji.prg, CompiledRunner) for ji in jit_cache): raise GraphException

    prgs = '\n'.join(dedup([cast(CompiledRunner, ji.prg).p.src for ji in jit_cache]))
    args = [f"{render_dtype(x.dtype)}* arg{i}" for i,x in enumerate(input_rawbuffers)]
    args += sorted([f"int {v.expr}" for v in var_vals])
    code = ["void batched("+','.join(args)+") {"]
    for ji in jit_cache:
      args = []
      for buf in ji.bufs:
        assert buf is not None
        if buf in input_rawbuffers:
          args.append(f"arg{input_rawbuffers.index(buf)}")
        else:
          args.append(f"({render_dtype(buf.dtype)}*)0x{ctypes.addressof(buf._buf):X}")
      args += [x.expr for x in cast(CompiledRunner, ji.prg).p.vars]
      code.append(f"  {cast(CompiledRunner, ji.prg).p.function_name}({','.join(args)});")
    code.append("}")
    if DEBUG >= 4: print("\n".join(code))
    compiler = Device["CPU"].compiler
    assert compiler is not None
    self._prg = ClangProgram("batched", compiler.compile(prgs+"\n"+"\n".join(code))) # no point in caching the pointers

  def __call__(self, rawbufs: List[Buffer], var_vals: Dict[Variable, int], wait=False):
    return cpu_time_execution(
    lambda: self._prg(*[x._buf for x in rawbufs], *[x[1] for x in sorted(var_vals.items(), key=lambda x: x[0].expr)]), enable=wait)