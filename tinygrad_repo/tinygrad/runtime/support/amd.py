import functools, tinygrad.runtime.autogen.am
from dataclasses import dataclass
from tinygrad.helpers import getbits

@dataclass
class AMDReg:
  name:str; offset:int; segment:int; fields:dict[str, tuple[int, int]]; bases:dict[int, tuple[int, ...]] # noqa: E702
  def __post_init__(self): self.addr:dict[int, int] = { inst: bases[self.segment] + self.offset for inst, bases in self.bases.items() }

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

  def fields_mask(self, *names) -> int:
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0]+1)) - 1) << self.fields[nm][0]) for nm in names), 0)

@dataclass
class AMDIP:
  name:str; version:tuple[int, int, int]; bases:dict[int, tuple[int, ...]] # noqa: E702

  @functools.cached_property
  def regs(self): return import_asic_regs(self.name, self.version, cls=functools.partial(AMDReg, bases=self.bases))

  def __getattr__(self, name:str):
    if name in self.regs: return self.regs[name]
    if (name10:=name.replace('reg', 'mm')) in self.regs: return self.regs[name10]
    raise AttributeError(f"{self.name.upper()} has no register {name}")

# load the greatest module with matching major version that's less than or equal to the target version
# this is not universally correct, see below for an example, but appears reliable for most recent gpus
# https://github.com/torvalds/linux/blob/9207d47f966be9f4d52e7e0119ac2b7a7e366f3e/drivers/gpu/drm/amd/amdgpu/amdgpu_discovery.c#L3163
def import_module(name:str, target:tuple[int, int, int], submod=""):
  # version overrides
  target = {("smu", (13, 0, 7)): (13, 0, 0)}.get((name, target), target)
  mod = getattr(tinygrad.runtime.autogen.am, submod) if submod else tinygrad.runtime.autogen.am
  if (children:=[c for c in mod.__all__ if c.startswith(name) and (v:=tuple(map(int, c.split('_')[1:])))[0] == target[0] and v <= target]):
    return getattr(mod, children[-1])
  raise ImportError(f"Failed to import {submod+'.' if submod else ''}{name} {'.'.join(map(str, target))}")

def import_soc(ip): return getattr(tinygrad.runtime.autogen.am, f"soc_{ip[0]}")

def import_pmc(ip) -> dict[str, tuple[str, int]]:
  from tinygrad.runtime.autogen.am import pmc
  # NOTE: precise arch for mi300+, generic for others, since rocm headers lack some archs
  return {k:x for k,v in pmc.counters.items() if (x:=v.get(f"gfx{ip[0]}{ip[1]:x}{ip[2]:x}" if ip[0] == 9 else f"gfx{ip[0]}", None)) is not None}

def import_asic_regs(prefix:str, version:tuple[int, int, int], cls=AMDReg) -> dict[str, AMDReg]:
  return {reg:cls(name=reg, offset=off, segment=seg, fields=fields) for reg,(off,seg,fields) in import_module(prefix, version, submod="regs").items()}
