import functools, importlib
from collections import defaultdict
from dataclasses import dataclass
from math import log2
from tinygrad.helpers import getbits

@dataclass(frozen=True)
class AMDRegBase:
  name: str
  offset: int
  segment: int
  fields: dict[str, tuple[int, int]]
  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

def collect_registers(module, cls=AMDRegBase) -> dict[str, AMDRegBase]:
  def _split_name(name): return name[:(pos:=next((i for i,c in enumerate(name) if c.isupper()), len(name)))], name[pos:]
  offsets = {k:v for k,v in module.__dict__.items() if _split_name(k)[0] in {'reg', 'mm'} and not k.endswith('_BASE_IDX')}
  bases = {k[:-len('_BASE_IDX')]:v for k,v in module.__dict__.items() if _split_name(k)[0] in {'reg', 'mm'} and k.endswith('_BASE_IDX')}
  fields: defaultdict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
  for field_name,field_mask in module.__dict__.items():
    if not ('__' in field_name and field_name.endswith('_MASK')): continue
    reg_name, reg_field_name = field_name[:-len('_MASK')].split('__')
    fields[reg_name][reg_field_name.lower()] = (int(log2(field_mask & -field_mask)), int(log2(field_mask)))
  # NOTE: Some registers like regGFX_IMU_FUSESTRAP in gc_11_0_0 are missing base idx, just skip them
  return {reg:cls(name=reg, offset=off, segment=bases[reg], fields=fields[_split_name(reg)[1]]) for reg,off in offsets.items() if reg in bases}

def import_module(name:str, version:tuple[int, ...], version_prefix:str=""):
  for ver in [version, version[:2]+(0,), version[:1]+(0, 0)]:
    try: return importlib.import_module(f"tinygrad.runtime.autogen.am.{name}_{version_prefix}{'_'.join(map(str, ver))}")
    except ImportError: pass
  raise ImportError(f"Failed to load autogen module for {name.upper()} {'.'.join(map(str, version))}")
