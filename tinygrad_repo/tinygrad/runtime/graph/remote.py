from tinygrad.ops import Variable
from tinygrad.engine.jit import GraphRunner
from tinygrad.engine.realize import CompiledRunner, ExecItem
from tinygrad.device import Device, Buffer
from tinygrad.runtime.ops_remote import GraphComputeItem, GraphAlloc, GraphFree, GraphExec
from tinygrad.helpers import unwrap, flatten, dedup, all_same

class RemoteGraph(GraphRunner):
  def __init__(self, jit_cache: list[ExecItem], rawbufs: list[Buffer], var_vals: dict[Variable, int]):
    super().__init__(jit_cache, rawbufs, var_vals)
    self.devices = dedup(flatten([[Device[unwrap(buf).device] for buf in ji.bufs] for ji in jit_cache]))
    assert all_same(self.devices), self.devices
    self.iids = sorted(self.input_replace.values())
    def _process_ji(ji: ExecItem):
      assert isinstance(ji.prg, CompiledRunner), f'Only compiled runners are supported: {ji.prg}'
      return GraphComputeItem(ji.prg._prg.name, ji.prg._prg.datahash, tuple(unwrap(buf)._buf for buf in ji.bufs), tuple(ji.prg.p.vars),
                              tuple(ji.prg.p.global_size) if ji.prg.p.global_size is not None else None,
                              tuple(ji.prg.p.local_size) if ji.prg.p.local_size is not None else None)
    self.graph_num = self.devices[0].graph_num
    self.devices[0].graph_num += 1
    self.devices[0].q(GraphAlloc(self.graph_num, tuple(_process_ji(ji) for ji in jit_cache), tuple(rawbufs[i]._buf for i in self.iids), var_vals))

  def __del__(self):
    self.devices[0].q(GraphFree(self.graph_num))

  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False):
    self.devices[0].q(GraphExec(self.graph_num, tuple(rawbufs[i]._buf for i in self.iids), var_vals, wait))
    if wait: return float(self.devices[0].batch_submit())
