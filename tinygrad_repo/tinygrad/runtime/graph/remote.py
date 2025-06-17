from tinygrad.uop.ops import Variable
from tinygrad.engine.jit import GraphRunner, MultiGraphRunner, GraphException
from tinygrad.engine.realize import CompiledRunner, BufferXfer, ExecItem
from tinygrad.device import Device, Buffer
from tinygrad.runtime.ops_remote import RemoteDevice, GraphComputeItem, Transfer, GraphAlloc, GraphFree, GraphExec
from tinygrad.helpers import unwrap, flatten, dedup, all_same
from typing import cast

class RemoteGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], rawbufs: list[Buffer], var_vals: dict[Variable, int]):
    super().__init__(jit_cache, rawbufs, var_vals)
    self.devices = dedup(flatten([[Device[unwrap(buf).device] for buf in ji.bufs] for ji in jit_cache]))
    if not all_same([d.conn for d in self.devices]): raise GraphException(f"Cross-host remote graph is not supported ({self.devices})")
    self.iids = sorted(self.input_replace.values())
    def _process_ji(ji: ExecItem):
      match ji.prg:
        case CompiledRunner():
          return GraphComputeItem(ji.prg.dev.session, ji.prg._prg.name, ji.prg._prg.datahash, tuple(unwrap(buf)._buf for buf in ji.bufs),
                                  tuple(ji.prg.p.vars), ji.fixedvars, tuple(ji.prg.p.ins), tuple(ji.prg.p.outs),
                                  tuple(ji.prg.p.global_size) if ji.prg.p.global_size is not None else None,
                                  tuple(ji.prg.p.local_size) if ji.prg.p.local_size is not None else None)
        case BufferXfer():
          dest, src = ji.bufs[0:2]
          assert dest is not None and src is not None, ji
          return Transfer(session=cast(RemoteDevice, Device[dest.device]).session, buffer_num=dest._buf,
                          ssession=cast(RemoteDevice, Device[src.device]).session, sbuffer_num=src._buf)
    self.graph_num = next(self.devices[0].graph_num)
    self.devices[0].q(GraphAlloc(self.graph_num, tuple(_process_ji(ji) for ji in jit_cache), self.map_rawbufs(rawbufs), var_vals))

  def __del__(self):
    self.devices[0].q(GraphFree(self.graph_num))

  def map_rawbufs(self, rawbufs:list[Buffer]):
    return tuple((cast(RemoteDevice, Device[rawbufs[i].device]).session, rawbufs[i]._buf) for i in self.iids)

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False):
    self.devices[0].q(GraphExec(self.graph_num, self.map_rawbufs(rawbufs), var_vals, wait))
    if wait: return float(self.devices[0].conn.batch_submit())

class RemoteMultiGraph(RemoteGraph, MultiGraphRunner): pass
