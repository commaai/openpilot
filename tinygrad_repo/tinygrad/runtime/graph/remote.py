import time, itertools
from tinygrad.uop.ops import Variable
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.engine.realize import CompiledRunner, BufferXfer, ExecItem
from tinygrad.device import Device, Compiled, Buffer
from tinygrad.runtime.ops_remote import RemoteDevice, RemoteConnection, RemoteRequest, GraphComputeItem, Transfer, GraphAlloc, GraphFree, GraphExec
from tinygrad.runtime.ops_remote import BatchTransfer, Event, Wait
from tinygrad.helpers import unwrap, flatten, dedup
from enum import Enum, auto
from dataclasses import replace
from collections import defaultdict
from typing import cast

class StagingType(Enum): NONE = auto(); GRAPH = auto(); TRANSFER = auto() # noqa: E702

def rd(dev:Compiled) -> RemoteDevice: return cast(RemoteDevice, dev)
def dev_key(dev:RemoteDevice): return dev.conn if dev.properties.graph_supports_multi else dev
def map_rawbuf(rawbuf:Buffer): return (cast(RemoteDevice, Device[rawbuf.device]).session, rawbuf._buf)

class RemoteGraph(MultiGraphRunner):
  def __init__(self, jit_cache: list[ExecItem], rawbufs: list[Buffer], var_vals: dict[Variable, int]):
    super().__init__(jit_cache, rawbufs, var_vals)
    devices = dedup(flatten([[Device[unwrap(buf).device] for buf in ji.bufs] for ji in jit_cache]))
    c2d = {device.conn: device for device in devices}
    self.handle_indexes = {map_rawbuf(rawbufs[i]): i for i in sorted(dedup(self.input_replace.values()))}

    self.template: list[RemoteRequest] = []

    stagings: dict[RemoteDevice|RemoteConnection, list[GraphComputeItem|Transfer]] = defaultdict(list)
    clobbered_buffers: set[Buffer] = set()
    cur_staging_type: StagingType = StagingType.NONE

    def _flush(new_staging_type:StagingType, force_break:bool=False):
      nonlocal cur_staging_type
      if cur_staging_type == new_staging_type and not force_break: return
      # Pre-sync
      if cur_staging_type == StagingType.TRANSFER:
        for sdev,ddev in itertools.permutations(c2d.values(), 2):
          self.template.append(Event(ddev.session, event:=next(ddev.event_num), session=sdev.session))
          self.template.append(Wait(event, session=ddev.session))
      # Flush
      for dev in devices:
        dk = dev_key(dev)
        staging = stagings[dk]
        if not staging: continue
        match cur_staging_type:
          case StagingType.GRAPH:
            bufs = tuple(map_rawbuf(rawbufs[i]) for i in sorted(dedup(self.input_replace.values())) if dev_key(rd(Device[rawbufs[i].device])) == dk)
            dev.q(GraphAlloc(graph_num:=next(dev.graph_num), tuple(staging), tuple(bufs), var_vals))
            self.template.append(GraphExec(graph_num, bufs, var_vals, wait=False, session=dev.session))
          case StagingType.TRANSFER:
            st = cast(list[Transfer], staging)
            for host in dedup(t.dsession.host for t in st):
              sbuffer_nums = [(unwrap(t.session), t.buffer_num) for t in st if t.dsession.host == host]
              dbuffer_nums = [(t.dsession, t.dbuffer_num) for t in st if t.dsession.host == host]
              self.template.append(BatchTransfer(sbuffer_nums, dbuffer_nums, session=dev.session))
        staging.clear()
      # Post-sync
      if cur_staging_type == StagingType.TRANSFER:
        for sdev,ddev in itertools.permutations(c2d.values(), 2):
          self.template.append(Event(ddev.session, event:=next(ddev.event_num), session=sdev.session))
          self.template.append(Wait(event, session=ddev.session))
      cur_staging_type = new_staging_type
      clobbered_buffers.clear()

    for ji in jit_cache:
      match ji.prg:
        case CompiledRunner():
          _flush(StagingType.GRAPH)
          gi = GraphComputeItem(ji.prg.dev.session, ji.prg._prg.name, ji.prg._prg.datahash, tuple(unwrap(buf)._buf for buf in ji.bufs),
                                tuple(ji.prg.p.vars), ji.fixedvars, tuple(ji.prg.p.ins), tuple(ji.prg.p.outs),
                                tuple(ji.prg.p.global_size) if ji.prg.p.global_size is not None else None,
                                tuple(ji.prg.p.local_size) if ji.prg.p.local_size is not None else None)
          stagings[dev_key(ji.prg.dev)].append(gi)
        case BufferXfer():
          dest, src = ji.bufs[0:2]
          dest_dev, src_dev = cast(RemoteDevice, Device[unwrap(dest).device]), cast(RemoteDevice, Device[unwrap(src).device])
          assert dest is not None and src is not None, ji
          ti = Transfer(session=src_dev.session, buffer_num=src._buf, dsession=dest_dev.session, dbuffer_num=dest._buf)
          if dev_key(dest_dev) == dev_key(src_dev):
            _flush(StagingType.GRAPH)
            stagings[dev_key(src_dev)].append(ti)
          elif dest_dev.conn == src_dev.conn:
            _flush(StagingType.NONE)
            self.template.append(ti)
          else:
            _flush(StagingType.TRANSFER, force_break=src in clobbered_buffers)
            clobbered_buffers.add(dest)
            stagings[dev_key(src_dev)].append(ti)
        case _: raise NotImplementedError(ji.prg)
    _flush(StagingType.NONE)
  def __del__(self):
    for req in self.template:
      match req:
        case GraphExec(): RemoteConnection(unwrap(req.session).host).q(GraphFree(req.graph_num, session=req.session))
  def __call__(self, rawbufs: list[Buffer], var_vals: dict[Variable, int], wait=False):
    if wait: st = time.perf_counter()
    rmap = {orig: map_rawbuf(rawbufs[replace_idx]) for orig,replace_idx in self.handle_indexes.items()}
    for req in self.template:
      match req:
        case GraphExec():
          req = replace(req, bufs=tuple(rmap[buf] for buf in req.bufs), var_vals=var_vals, wait=wait)
        case Transfer():
          if (req.session, req.buffer_num) in rmap: req = replace(req, buffer_num=rmap[(req.session, req.buffer_num)][1])
          if (req.dsession, req.dbuffer_num) in rmap: req = replace(req, dbuffer_num=rmap[(req.dsession, req.dbuffer_num)][1])
        case BatchTransfer():
          req = replace(req, sbuffer_nums=[rmap.get(b, b) for b in req.sbuffer_nums], dbuffer_nums=[rmap.get(b, b) for b in req.dbuffer_nums])
        case Event()|Wait():
          pass # event number can be reused
        case _: raise NotImplementedError(req)
      RemoteConnection(unwrap(req.session).host).q(req)
    if wait:
      RemoteConnection(unwrap(req.session).host).batch_submit()
      return time.perf_counter() - st
