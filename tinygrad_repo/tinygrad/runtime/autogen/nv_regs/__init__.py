import re, pathlib
from tinygrad.runtime.autogen import load, nv_src

swref_path, hwref_path = "{}/src/common/inc/swref/published", "{}/kernel-open/nvidia-uvm/hwref"
swref = {
  "dev_therm": ["gb202"], "dev_vm": ["tu102", "gh100"], **{k:["tu102"] for k in ["dev_fb", "dev_bus"]},
  **{k:["ga102"] for k in ["dev_gc6_island", "dev_gsp", "dev_riscv_pri", "dev_fbif_v4", "dev_falcon_second_pri", "dev_sec_pri"]},
  "dev_falcon_v4": ["ga102", "gh100"], "dev_fsp_pri": ["gh100"]
}
hwref = {"dev_mmu": ["tu102", "gh100"]}
__all__ = ["nv_ref", *swref.keys(), *hwref.keys()]
has_addendum = (("ga102", "dev_gc6_island"), ("ga102", "dev_falcon_v4"))

def __getattr__(nm):
  arch_map = {"tu102":"turing", "ga102":"ampere", "gh100":"hopper", "gb202":"blackwell"}
  regs_off = {'NV_PFALCON_FALCON': 0x0, 'NV_PGSP_FALCON': 0x0, 'NV_PSEC_FALCON': 0x0, 'NV_PRISCV_RISCV': 0x1000, 'NV_PGC6_AON': 0x0, 'NV_PFSP': 0x0,
    'NV_PGC6_BSI': 0x0, 'NV_PFALCON_FBIF': 0x600, 'NV_PFALCON2_FALCON': 0x1000, 'NV_PBUS': 0x0, 'NV_PFB': 0x0, 'NV_PMC': 0x0, 'NV_PGSP_QUEUE': 0x0,
    'NV_VIRTUAL_FUNCTION':0xb80000, "NV_THERM": 0x0}

  def genreg(_, files, **kwargs):
    out = []
    for (file, arch) in [(file, "" if (a:=file.split('/')[-2]) == "published" else a) for file in files]:
      lines = ((p:=pathlib.Path(file)).read_text() + ((p.parent/f"{nm}_addendum.h").read_text() if (arch, nm) in has_addendum else "")).splitlines()
      def extract(pat): return (m.groups() for l in lines if (m:=re.match(pat, l)))
      bitfields = {k:f"({lo}, {hi})" for k,hi,lo in extract(r'#define\s+(\w+)\s+([0-9\+\-\*\(\)]+):([0-9\+\-\*\(\)]+)')}
      regs = {}
      for l in lines:
        def off(name): return next((o for p,o in regs_off.items() if name.startswith(p)), None)
        def fields(name): return "{" + ", ".join(f"{k[len(name)+1:].lower()!r}: {v}" for k, v in bitfields.items() if k.startswith(name+"_")) + "}"
        if (m:=re.match(r'#define\s+(\w+)\s*\(\s*(\w+)\s*\)\s*(.+)', l)) and off(m.group(1)) is not None:
          regs[m.group(1)] = f"(0x{off(m.group(1)):X}, lambda {m.group(2)}: " + re.sub(r' */\*.*\*/', '', m.group(3)) + f", {fields(m.group(1))})"
        elif (m:=re.match(r'#define\s+(\w+)\s+([0-9A-Fa-fx]+)(?![^\n]*:)', l)):
          if off(m.group(1)) is None or any(m.group(1).startswith(r+'_') for r in regs): regs[m.group(1)] = m.group(2)
          else: regs[m.group(1)] = f"(0x{off(m.group(1)):X}, {m.group(2)}, {fields(m.group(1))})"
        elif (m:=re.match(r'#define\s+(\w+)\s*/\* ----G \*/\s*$', l)): regs[m.group(1)] = f"(None, None, {fields(m.group(1))})" # groups (for MMU)
      out.extend([f"{arch or 'regs'} = {{", *[f"  {k!r}: {v}," for k,v in regs.items()], "}"])
    return "\n".join(out)

  if nm == "nv_ref": return load(f"nv_regs/{nm}", [f"{swref_path}/{nm}.h"], gen=genreg, srcs=nv_src["nv_570"])
  if nm in __all__:
    path, arches = (swref_path, swref[nm]) if nm in swref else (hwref_path, hwref[nm])
    return load(f"nv_regs/{nm}", [f"{path}/{arch_map[arch]}/{arch}/{nm}.h" for arch in arches], gen=genreg, srcs=nv_src["nv_570"])
  raise AttributeError(f"no such autogen: {nm}")
