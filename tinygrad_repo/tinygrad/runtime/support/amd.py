import functools, importlib, re, urllib
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.helpers import getbits, fetch

AMDGPU_URL = "https://gitlab.com/linux-kernel/linux-next/-/raw/cf6d949a409e09539477d32dbe7c954e4852e744/drivers/gpu/drm/amd"
ROCM_URL = "https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/projects"

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
  name:str; version:tuple[int, ...]; bases:dict[int, tuple[int, ...]] # noqa: E702
  def __post_init__(self): self.version = fixup_ip_version(self.name, self.version)[0]

  @functools.cached_property
  def regs(self): return import_asic_regs(self.name, self.version, cls=functools.partial(AMDReg, bases=self.bases))

  def __getattr__(self, name:str):
    if name in self.regs: return self.regs[name]
    if (name10:=name.replace('reg', 'mm')) in self.regs: return self.regs[name10]
    raise AttributeError(f"{self.name.upper()} has no register {name}")

def fixup_ip_version(ip:str, version:tuple[int, ...]) -> list[tuple[int, ...]]:
  # override versions
  def _apply_ovrd(ovrd:dict[tuple[int, ...], tuple[int, ...]]) -> tuple[int, ...]:
    for ver, ovrd_ver in ovrd.items():
      if version[:len(ver)] == ver: return ovrd_ver
    return version

  if ip in ['nbio', 'nbif']: version = _apply_ovrd({(3,3): (2,3,0), (7,3): (7,2,0)})
  elif ip in ['mp', 'smu']: version = _apply_ovrd({(14,0,3): (14,0,2)})
  elif ip in ['gc']: version = _apply_ovrd({(9,5,0): (9,4,3)})

  return [version, version[:2], version[:2]+(0,), version[:1]+(0, 0)]

def header_download(file, name=None, subdir="defines", url=AMDGPU_URL) -> str: return fetch(f"{url}/{file}", name=name, subdir=subdir).read_text()

def import_header(path:str, url=AMDGPU_URL):
  t = re.sub(r'//.*|/\*.*?\*/','', header_download(path, subdir="defines", url=url), flags=re.S)
  # TODO: refactor when clang2py is replaced
  return {k:int(v,0) for k,v in re.findall(r'\b([A-Za-z_]\w*)\s*=\s*(0x[0-9A-Fa-f]+|\d+)', t) + \
                                re.findall(r'^\s*#\s*define\s+([A-Za-z_0-9]\w*)\s+(0x[0-9A-Fa-f]+|\d+)', t, re.M)}

def import_module(name:str, version:tuple[int, ...], version_prefix:str=""):
  for ver in fixup_ip_version(name, version):
    try: return importlib.import_module(f"tinygrad.runtime.autogen.am.{name}_{version_prefix}{'_'.join(map(str, ver))}")
    except ImportError: pass
  raise ImportError(f"Failed to load autogen module for {name.upper()} {'.'.join(map(str, version))}")

def import_soc(ip):
  # rocm soc headers have more profiling enums than upstream linux
  return type("SOC", (object,), import_header(f"aqlprofile/linux/{({9: 'vega10', 10: 'navi10', 11: 'soc21', 12: 'soc24'}[ip[0]])}_enum.h", ROCM_URL))

def import_ip_offsets(ip): return type("IPOFF", (object,), import_header(f"include/{('sienna_cichlid' if ip[0] > 9 else 'vega20')}_ip_offset.h"))

def import_pmc(ip) -> dict[str, tuple[str, int]]:
  res:dict[str, tuple[str, int]] = {}
  arch = f"gfx{ip[0]}{ip[1]:x}{ip[2]:x}"

  for sec in header_download("rocprofiler-compute/src/rocprof_compute_soc/profile_configs/counter_defs.yaml", url=ROCM_URL).split('- name: ')[1:]:
    for arch_spec in sec.split('- architectures:')[1:]:
      if arch in arch_spec and (block:=re.search(r'block:\s*([A-Za-z0-9_]+)', arch_spec)) and (ev:=re.search(r'event:\s*(\d+)', arch_spec)):
        res[sec.splitlines()[0].strip()] = (block.group(1), int(ev.group(1)))

  return res

def import_asic_regs(prefix:str, version:tuple[int, ...], cls=AMDReg) -> dict[str, AMDReg]:
  def _split_name(name): return name[:(pos:=next((i for i,c in enumerate(name) if c.isupper()), len(name)))], name[pos:]
  def _extract_regs(txt):
    return {m.group(1): int(m.group(2), 0) for line in txt.splitlines() if (m:=re.match(r'#define\s+(\S+)\s+(0x[\da-fA-F]+|\d+)', line))}
  def _download_file(ver, suff) -> str:
    dir_prefix = {"osssys": "oss"}.get(prefix, prefix)
    fetch_name, file_name = f"{prefix}_{'_'.join(map(str, ver))}_{suff}.h", f"{prefix}_{'_'.join(map(str, version))}_{suff}.h"
    return header_download(f"include/asic_reg/{dir_prefix}/{fetch_name}", name=file_name, subdir="asic_regs")

  for ver in fixup_ip_version(prefix, version):
    try: offs, sh_masks = _extract_regs(_download_file(ver, "offset")), _extract_regs(_download_file(ver, "sh_mask"))
    except urllib.error.HTTPError as e:
      if e.code == 404: continue
      raise

    offsets = {k:v for k,v in offs.items() if _split_name(k)[0] in {'reg', 'mm'} and not k.endswith('_BASE_IDX')}
    bases = {k[:-len('_BASE_IDX')]:v for k,v in offs.items() if _split_name(k)[0] in {'reg', 'mm'} and k.endswith('_BASE_IDX')}

    fields: defaultdict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for field_name, field_mask in sh_masks.items():
      if not ('__' in field_name and field_name.endswith('_MASK')): continue
      reg_name, reg_field_name = field_name[:-len('_MASK')].split('__')
      fields[reg_name][reg_field_name.lower()] = ((field_mask & -field_mask).bit_length()-1, field_mask.bit_length()-1)

    # NOTE: Some registers like regGFX_IMU_FUSESTRAP in gc_11_0_0 are missing base idx, just skip them
    return {reg:cls(name=reg, offset=off, segment=bases[reg], fields=fields[_split_name(reg)[1]]) for reg,off in offsets.items() if reg in bases}
  raise ImportError(f"Failed to load ASIC registers for {prefix.upper()} {'.'.join(map(str, version))}")
