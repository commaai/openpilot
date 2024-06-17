#!/usr/bin/env python3
import os
import re
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = HERE + "/.."

# blacklisting is for two purposes:
# - minimizing release download size
# - keeping the diff readable
blacklist = [
  "body/STL/",

  "panda/drivers/",
  "panda/examples/",
  "panda/tests/safety/",

  "opendbc/.*.dbc$",
  "opendbc/generator/",

  "cereal/.*test.*",
  "^common/tests/",

  # particularly large text files
  "poetry.lock",
  "third_party/catch2",
  "selfdrive/car/tests/test_models.*",

  "^tools/",
  "^scripts/",
  "^tinygrad_repo/",

  "matlab.*.md",

  ".git/",
  ".github/",
  ".devcontainer/",
  "Darwin/",
  ".vscode",

  # common things
  "LICENSE",
  "Dockerfile",
  ".pre-commit",

  # no LFS or submodules in release
  ".lfsconfig",
  ".gitattributes",
  ".git$",
  ".gitmodules",
]

# gets you through the blacklist
whitelist = [
  "tools/lib/",
  "tools/bodyteleop/",

  "tinygrad_repo/openpilot/compile2.py",
  "tinygrad_repo/extra/onnx.py",
  "tinygrad_repo/extra/onnx_ops.py",
  "tinygrad_repo/extra/thneed.py",
  "tinygrad_repo/extra/utils.py",
  "tinygrad_repo/tinygrad/codegen/kernel.py",
  "tinygrad_repo/tinygrad/codegen/linearizer.py",
  "tinygrad_repo/tinygrad/features/image.py",
  "tinygrad_repo/tinygrad/features/search.py",
  "tinygrad_repo/tinygrad/nn/*",
  "tinygrad_repo/tinygrad/renderer/cstyle.py",
  "tinygrad_repo/tinygrad/renderer/opencl.py",
  "tinygrad_repo/tinygrad/runtime/lib.py",
  "tinygrad_repo/tinygrad/runtime/ops_cpu.py",
  "tinygrad_repo/tinygrad/runtime/ops_disk.py",
  "tinygrad_repo/tinygrad/runtime/ops_gpu.py",
  "tinygrad_repo/tinygrad/shape/*",
  "tinygrad_repo/tinygrad/.*.py",

  # TODO: do this automatically
  "opendbc/comma_body.dbc",
  "opendbc/chrysler_ram_hd_generated.dbc",
  "opendbc/chrysler_ram_dt_generated.dbc",
  "opendbc/chrysler_pacifica_2017_hybrid_generated.dbc",
  "opendbc/chrysler_pacifica_2017_hybrid_private_fusion.dbc",
  "opendbc/gm_global_a_powertrain_generated.dbc",
  "opendbc/gm_global_a_object.dbc",
  "opendbc/gm_global_a_chassis.dbc",
  "opendbc/FORD_CADS.dbc",
  "opendbc/ford_fusion_2018_adas.dbc",
  "opendbc/ford_lincoln_base_pt.dbc",
  "opendbc/honda_accord_2018_can_generated.dbc",
  "opendbc/acura_ilx_2016_can_generated.dbc",
  "opendbc/acura_rdx_2018_can_generated.dbc",
  "opendbc/acura_rdx_2020_can_generated.dbc",
  "opendbc/honda_civic_touring_2016_can_generated.dbc",
  "opendbc/honda_civic_hatchback_ex_2017_can_generated.dbc",
  "opendbc/honda_crv_touring_2016_can_generated.dbc",
  "opendbc/honda_crv_ex_2017_can_generated.dbc",
  "opendbc/honda_crv_ex_2017_body_generated.dbc",
  "opendbc/honda_crv_executive_2016_can_generated.dbc",
  "opendbc/honda_fit_ex_2018_can_generated.dbc",
  "opendbc/honda_odyssey_exl_2018_generated.dbc",
  "opendbc/honda_odyssey_extreme_edition_2018_china_can_generated.dbc",
  "opendbc/honda_insight_ex_2019_can_generated.dbc",
  "opendbc/acura_ilx_2016_nidec.dbc",
  "opendbc/honda_civic_ex_2022_can_generated.dbc",
  "opendbc/hyundai_canfd.dbc",
  "opendbc/hyundai_kia_generic.dbc",
  "opendbc/hyundai_kia_mando_front_radar_generated.dbc",
  "opendbc/mazda_2017.dbc",
  "opendbc/nissan_x_trail_2017_generated.dbc",
  "opendbc/nissan_leaf_2018_generated.dbc",
  "opendbc/subaru_global_2017_generated.dbc",
  "opendbc/subaru_global_2020_hybrid_generated.dbc",
  "opendbc/subaru_outback_2015_generated.dbc",
  "opendbc/subaru_outback_2019_generated.dbc",
  "opendbc/subaru_forester_2017_generated.dbc",
  "opendbc/toyota_tnga_k_pt_generated.dbc",
  "opendbc/toyota_new_mc_pt_generated.dbc",
  "opendbc/toyota_nodsu_pt_generated.dbc",
  "opendbc/toyota_adas.dbc",
  "opendbc/toyota_tss2_adas.dbc",
  "opendbc/vw_golf_mk4.dbc",
  "opendbc/vw_mqb_2010.dbc",
  "opendbc/tesla_can.dbc",
  "opendbc/tesla_radar_bosch_generated.dbc",
  "opendbc/tesla_radar_continental_generated.dbc",
  "opendbc/tesla_powertrain.dbc",
]


if __name__ == "__main__":
  for f in Path(ROOT).rglob("**/*"):
    if not (f.is_file() or f.is_symlink()):
      continue

    rf = str(f.relative_to(ROOT))
    blacklisted = any(re.search(p, rf) for p in blacklist)
    whitelisted = any(re.search(p, rf) for p in whitelist)
    if blacklisted and not whitelisted:
      continue

    print(rf)
