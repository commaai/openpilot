import pathlib, hashlib, re, itertools
from tinygrad.runtime.autogen import load, root

__all__ = ["am", "pm4_soc15", "pm4_nv", "sdma_4_0_0", "sdma_5_0_0", "sdma_6_0_0", "smu_13_0_0", "smu_13_0_6", "smu_13_0_12", "smu_14_0_2",
           "fw", "navi_offsets", "vega_offsets", "regs", "soc_9", "soc_11", "soc_12", "pmc"]

am_src="https://github.com/ROCm/ROCK-Kernel-Driver/archive/33970e1351f5e511029602454979f3de7e22260f.tar.gz"
rocm_src="https://github.com/ROCm/rocm-systems/archive/cccc350dc620e61ae2554978b62ab3532dc10bd9.tar.gz"
AMD, AMDINC = "{}/drivers/gpu/drm/amd", "{}/drivers/gpu/drm/amd/include"
inc, kern_rules = ["-include", "stdint.h"], [(r'le32_to_cpu', ''),]
fw_src="https://gitlab.com/kernel-firmware/linux-firmware/-/archive/1e2c15348485939baf1b6d1f5a7a3b799d80703d/1e2c15348485939baf1b6d1f5a7a3b799d80703d.tar.gz"
pmc_src="https://raw.githubusercontent.com/ROCm/rocm-systems/cccc350dc620e61ae2554978b62ab3532dc10bd9/projects/rocprofiler-compute/src/rocprof_compute_soc/profile_configs/counter_defs.yaml"

reg_files = {
  "gc": [(9,4,3), (11,0,0), (11,0,3), (11,5,0), (12,0,0)],
  "mmhub": [(1,8,0), (3,0,0), (3,0,1), (3,0,2), (3,3,0), (4,1,0)],
  "nbio": [(4,3,0), (7,2,0), (7,7,0), (7,9,0), (7,11,0)], "nbif": [(6,3,1)],
  "mp": [(11,0,0), (13,0,0), (14,0,2)], "hdp": [(4,4,2), (6,0,0), (7,0,0)],
  "osssys": [(4,4,2), (6,0,0), (6,1,0), (7,0,0)], "sdma": [(4,4,2)]
}

reg_patterns = {
  "gc": ["GCVM", "GCMC_VM", "CP_(HQD|MQD|MEC|ME_CNTL|PERFMON|RB_WPTR_POLL_CNTL|INT_CNTL|STAT|PFP_PRGRM|ME_PRGRM|COHER_START)", "COMPUTE_",
         "(SQ|GL2C|TCC)_PERFCOUNTER", "SQ_THREAD_TRACE", "SPI_(CONFIG_CNTL|COMPUTE_QUEUE_RESET)", "GRBM", "SH_MEM", "RLC", "TCP", "GB_ADDR_CONFIG",
         "SDMA[01]_(WATCHDOG_CNTL|UTCL1_(CNTL|PAGE)|MCU_CNTL|F32_CNTL|CNTL|QUEUE0_|RLC_CGCG_CTRL)", "SCRATCH_REG[67]"],
  "mmhub": ["MMVM", "MMMC_VM", "MM_ATC_L2_MISC_CG"],
  "nbio": (nbio:=["BIF_BX_PF[01]_GPU_HDP_FLUSH", "BIF_BX_PF0_RSMU", "BIF_BX0_(REMAP_HDP_MEM_FLUSH_CNTL|BIF_DOORBELL_INT_CNTL|PCIE_INDEX2|PCIE_DATA2)",
                  "BIFC_(DOORBELL_ACCESS_EN_PF|GFX_INT_MONITOR_MASK)", "XCC_DOORBELL_FENCE", "DOORBELL0_CTRL_ENTRY", "GDC_S2A0_S2A_DOORBELL_ENTRY",
                  "S2A_DOORBELL_ENTRY", "RCC_DEV0_EPF0_RCC_DOORBELL_APER_EN", "RCC_DEV0_EPF2_STRAP2"]),
  "nbif": nbio,
  "mp": ["MP([01]|ASP)_SMN_C2PMSG"], "hdp": ["HDP_MEM_POWER_CTRL"], "oss": ["IH_"], "sdma": ["SDMA_GFX", "SDMA_CNTL"]
}

soc_patterns = ["SQ_TT", "VGT_EVENT_TYPE", "CS", "MTYPE", "SH"]

def __getattr__(nm):
  match nm:
    case "am": return load("am/am", [root/f"extra/amdpci/headers/{s}.h" for s in ["v11_structs", "v12_structs", "amdgpu_vm",
      "discovery", "amdgpu_ucode", "psp_gfx_if", "amdgpu_psp", "amdgpu_irq", "amdgpu_doorbell"]] + [f"{AMD}/amdkfd/soc15_int.h"] + \
      [f"{AMDINC}/ivsrcid/{s}.h" for s in [f"gfx/irqsrcs_gfx_{x}_0" for x in ('9','11_0','12_0')] + [f"sdma0/irqsrcs_sdma0_{x}_0" for x in (4,5)]] + \
      [f"{AMDINC}/{s}.h" for s in ["v9_structs", "soc15_ih_clientid"]], args=inc, srcs=am_src, rules=kern_rules)
    case "pm4_soc15": return load("am/pm4_soc15", [f"{AMD}/amdkfd/kfd_pm4_headers_ai.h", f"{AMD}/amdgpu/soc15d.h"], srcs=am_src)
    case "pm4_nv": return load("am/pm4_nv", [f"{AMD}/amdkfd/kfd_pm4_headers_ai.h", f"{AMD}/amdgpu/nvd.h"], srcs=am_src)
    case "sdma_4_0_0": return load("am/sdma_4_0_0", [root/"extra/hip_gpu_driver/sdma_registers.h", f"{AMD}/amdgpu/vega10_sdma_pkt_open.h"],
                                   args=["-I/opt/rocm/include", "-x", "c++"], srcs=am_src)
    case "sdma_5_0_0": return load("am/sdma_5_0_0", [root/"extra/hip_gpu_driver/sdma_registers.h", f"{AMD}/amdgpu/navi10_sdma_pkt_open.h"],
                                   args=["-I/opt/rocm/include", "-x", "c++"], srcs=am_src)
    case "sdma_6_0_0": return load("am/sdma_6_0_0", [root/"extra/hip_gpu_driver/sdma_registers.h", f"{AMD}/amdgpu/sdma_v6_0_0_pkt_open.h"],
                                   args=["-I/opt/rocm/include", "-x", "c++"], srcs=am_src)
    case "smu_13_0_0": return load("am/smu_13_0_0", [f"{AMD}/pm/swsmu/inc/pmfw_if/{s}.h" for s in ["smu_v13_0_0_ppsmc","smu13_driver_if_v13_0_0"]]
                                    +[root/"extra/amdpci/headers/amdgpu_smu.h"], args=inc, srcs=am_src)
    case "smu_13_0_6": return load("am/smu_13_0_6", [f"{AMD}/pm/swsmu/inc/pmfw_if/{s}.h" for s in ["smu_v13_0_6_ppsmc","smu_v13_0_6_pmfw", \
      "smu13_driver_if_v13_0_6"]] +[root/"extra/amdpci/headers/amdgpu_smu.h"], args=inc, srcs=am_src)
    case "smu_13_0_12": return load("am/smu_13_0_12", [f"{AMD}/pm/swsmu/inc/pmfw_if/{s}.h" for s in ["smu_v13_0_12_ppsmc","smu_v13_0_12_pmfw",
      "smu13_driver_if_v13_0_6"]] +[root/"extra/amdpci/headers/amdgpu_smu.h"], args=inc, srcs=am_src)
    case "smu_14_0_2": return load("am/smu_14_0_2", [f"{AMD}/pm/swsmu/inc/pmfw_if/{s}.h" for s in ["smu_v14_0_0_pmfw", "smu_v14_0_2_ppsmc",
                                    "smu14_driver_if_v14_0"]]+[root/"extra/amdpci/headers/amdgpu_smu.h"], args=inc, srcs=am_src)
    # firmware hashes
    case "fw":
      def genfw(name, files, **kwargs): return "\n".join(["hashes = {"] + [f"  {p.name!r}: {hashlib.sha256(p.read_bytes()).hexdigest()!r},"
                                                                           for f in files if (p:=pathlib.Path(f)).is_file()] + ["}"])
      return load("am/fw", ["{}/amdgpu/psp_*_sos.bin", "{}/amdgpu/smu_*.bin", "{}/amdgpu/sdma_*.bin"] +
                           [f"{{}}/amdgpu/gc_*_{x}.bin" for x in ["pfp", "me", "mec", "imu", "rlc"]], srcs=fw_src, gen=genfw)
    case "navi_offsets": return load("am/navi_offsets", [f"{AMD}/include/sienna_cichlid_ip_offset.h"], srcs=am_src)
    case "vega_offsets": return load("am/vega_offsets", [f"{AMD}/include/vega20_ip_offset.h"], srcs=am_src)
    case "regs":
      def genreg(_, files, **kwargs):
        out = ["__all__ = " + repr([file.split('/')[-1] for file in files])]
        for file, nm in [(file.replace("mp_11_0_0", "mp_11_0"), file.split('/')[-1]) for file in files]:
          pats = reg_patterns[prefix := {"osssys": "oss"}.get(x:=nm.split("_", 1)[0], x)]

          def split_name(name): return name[:(pos:=next((i for i,c in enumerate(name) if c.isupper()), len(name)))], name[pos:]
          # handle CDNA's different register names
          def normalize(reg):
            return s[0] + prefix.upper()[:2] + s[1] if prefix in ("gc", "mmhub") and (s:=split_name(reg))[1].startswith(("VM_", "MC_VM_")) else reg
          def extract(lines, pat): return ((normalize(m.group(1)), int(m.group(2), 0)) for l in lines if (m:=re.match(pat, l)))

          offset, sh_mask = pathlib.Path(f"{file}_offset.h").read_text().splitlines(), pathlib.Path(f"{file}_sh_mask.h").read_text().splitlines()
          defs = {k:v for k,v in extract(offset, r'#define\s+((?:mm|reg)\S+)\s+(0x[\da-fA-F]+|\d+)') if any(re.match("(mm|reg)"+p, k) for p in pats)}
          fields = {reg: {name.split('__')[1].lower(): ((mask & -mask).bit_length() - 1, mask.bit_length() - 1) for name, mask in fs}
                    for reg, fs in itertools.groupby(extract(sh_mask, r'#define\s+(\S+)_MASK\s+(0x[\da-fA-F]+|\d+)'), lambda x: x[0].split('__')[0])}

          regs = {reg: (off, defs[f"{reg}_BASE_IDX"], fields.get(split_name(reg)[1], {})) for reg,off in defs.items() if f"{reg}_BASE_IDX" in defs}
          print(f"defined {len(regs)} registers for {nm}")
          out.extend([f"{nm} = {{"] + [f"  {k!r}: {v!r}," for k,v in regs.items()] + ["}"])
        return "\n".join(out)
      return load("am/regs", [AMDINC + "/asic_reg/" + {"osssys":"oss"}.get(pre, pre) + f"/{pre}_{'_'.join(map(str, ver))}"
                              for pre in reg_files for ver in sorted(reg_files[pre])], srcs=am_src, gen=genreg)
    case "soc_9" | "soc_11" | "soc_12":
      return load(f"am/{nm}", ["{}/projects/aqlprofile/linux/" + {9: "vega10", 11: "soc21", 12: "soc24"}[int(nm.split('_')[1])] + "_enum.h"],
                  srcs=rocm_src, patterns=soc_patterns, macros=False)
    case "pmc":
      def genpmc(_, files, **kwargs):
        from yaml import safe_load # type: ignore
        with open(files[0], "r") as f: data = safe_load(f)
        out = ["counters = {"]
        for counter in [c for c in data['rocprofiler-sdk']['counters'] if any('block' in d for d in c['definitions'])]:
          out.extend([f"  {counter['name']!r}: {{",
                      *[f"    {a!r}: ({d['block']!r}, {d['event']})," for d in counter['definitions'] for a in d['architectures']], "  },"])
        return "\n".join(out + ["}"])
      return load("am/pmc", ["{}/counter_defs.yaml"], srcs=pmc_src, gen=genpmc)
    case _: raise AttributeError(f"no such autogen: {nm}")
