from tinygrad.runtime.autogen import load, root

am_src="https://github.com/ROCm/ROCK-Kernel-Driver/archive/33970e1351f5e511029602454979f3de7e22260f.tar.gz"
AMD, AMDINC = "{}/drivers/gpu/drm/amd", "{}/drivers/gpu/drm/amd/include"
inc, kern_rules = ["-include", "stdint.h"], [(r'le32_to_cpu', ''),]

def __getattr__(nm):
  match nm:
    case "am": return load("am/am", [], [root/f"extra/amdpci/headers/{s}.h" for s in ["v11_structs", "v12_structs", "amdgpu_vm",
      "discovery", "amdgpu_ucode", "psp_gfx_if", "amdgpu_psp", "amdgpu_irq", "amdgpu_doorbell"]] + [f"{AMD}/amdkfd/soc15_int.h"] + \
      [f"{AMDINC}/ivsrcid/{s}.h" for s in [f"gfx/irqsrcs_gfx_{x}_0" for x in ('9','11_0','12_0')] + [f"sdma0/irqsrcs_sdma0_{x}_0" for x in (4,5)]] + \
      [f"{AMDINC}/{s}.h" for s in ["v9_structs", "soc15_ih_clientid"]], args=inc, tarball=am_src, rules=kern_rules)
    case "pm4_soc15": return load("am/pm4_soc15", [], [f"{AMD}/amdkfd/kfd_pm4_headers_ai.h", f"{AMD}/amdgpu/soc15d.h"], tarball=am_src)
    case "pm4_nv": return load("am/pm4_nv", [], [f"{AMD}/amdkfd/kfd_pm4_headers_ai.h", f"{AMD}/amdgpu/nvd.h"], tarball=am_src)
    case "sdma_4_0_0": return load("am/sdma_4_0_0", [], [root/"extra/hip_gpu_driver/sdma_registers.h", f"{AMD}/amdgpu/vega10_sdma_pkt_open.h"],
                                   args=["-I/opt/rocm/include", "-x", "c++"], tarball=am_src)
    case "sdma_5_0_0": return load("am/sdma_5_0_0", [], [root/"extra/hip_gpu_driver/sdma_registers.h", f"{AMD}/amdgpu/navi10_sdma_pkt_open.h"],
                                   args=["-I/opt/rocm/include", "-x", "c++"], tarball=am_src)
    case "sdma_6_0_0": return load("am/sdma_6_0_0", [], [root/"extra/hip_gpu_driver/sdma_registers.h", f"{AMD}/amdgpu/sdma_v6_0_0_pkt_open.h"],
                                   args=["-I/opt/rocm/include", "-x", "c++"], tarball=am_src)
    case "smu_v13_0_0": return load("am/smu_v13_0_0",[],[f"{AMD}/pm/swsmu/inc/pmfw_if/{s}.h" for s in ["smu_v13_0_0_ppsmc","smu13_driver_if_v13_0_0"]]
                                    +[root/"extra/amdpci/headers/amdgpu_smu.h"], args=inc, tarball=am_src)
    case "smu_v13_0_6": return load("am/smu_v13_0_6",[],[f"{AMD}/pm/swsmu/inc/pmfw_if/{s}.h" for s in ["smu_v13_0_6_ppsmc","smu_v13_0_6_pmfw", \
      "smu13_driver_if_v13_0_6"]] +[root/"extra/amdpci/headers/amdgpu_smu.h"], args=inc, tarball=am_src)
    case "smu_v14_0_2": return load("am/smu_v14_0_2", [], [f"{AMD}/pm/swsmu/inc/pmfw_if/{s}.h" for s in ["smu_v14_0_0_pmfw", "smu_v14_0_2_ppsmc",
                                    "smu14_driver_if_v14_0"]]+[root/"extra/amdpci/headers/amdgpu_smu.h"], args=inc, tarball=am_src)
    case _: raise AttributeError(f"no such autogen: {nm}")
