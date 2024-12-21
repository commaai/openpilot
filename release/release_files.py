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
  "panda/drivers/",
  "panda/examples/",
  "panda/tests/safety/",

  "opendbc_repo/dbc/.*.dbc$",
  "opendbc_repo/dbc/generator/",

  "cereal/.*test.*",
  "^common/tests/",

  # particularly large text files
  "uv.lock",
  "third_party/catch2",
  "selfdrive/car/tests/test_models.*",

  "^tools/",
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

# Sunnypilot blacklist
sunnypilot_blacklist = [
  "system/loggerd/sunnylink_uploader.py",  # Temporarily, until we are ready to roll it out widely
  "system/manager/gitlab_runner.sh",
  ".idea/",
  ".run/",
  ".*__pycache__/.*",
  ".*\\.pyc",
  "teleoprtc/*",
  "third_party/snpe/x86_64/*",
  "body/board/canloader.py",
  "body/board/flash_base.sh",
  "body/board/flash_knee.sh",
  "body/board/recover.sh",
  ".*/test/",
  ".*/tests/",
  ".*tinygrad_repo/tinygrad/renderer/",
  "README.md",
  ".*internal/",
  "docs/.*",
  ".sconsign.dblite",
  "release/ci/scons_cache/",
  ".gitlab-ci.yml",
  ".clang-tidy",
  ".dockerignore",
  ".editorconfig",
  ".python-version",
  "SECURITY.md",
  "codecov.yml",
  "conftest.py",
  "poetry.lock",
  ".venv/",
]

# Merge the blacklists
blacklist += sunnypilot_blacklist

# gets you through the blacklist
whitelist = [
  "tools/lib/",
  "tools/bodyteleop/",
  "tools/joystick/",
  "tools/longitudinal_maneuvers/",

  "tinygrad_repo/examples/openpilot/compile3.py",
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
  "opendbc_repo/dbc/comma_body.dbc",
  "opendbc_repo/dbc/chrysler_ram_hd_generated.dbc",
  "opendbc_repo/dbc/chrysler_ram_dt_generated.dbc",
  "opendbc_repo/dbc/chrysler_pacifica_2017_hybrid_generated.dbc",
  "opendbc_repo/dbc/chrysler_pacifica_2017_hybrid_private_fusion.dbc",
  "opendbc_repo/dbc/gm_global_a_powertrain_generated.dbc",
  "opendbc_repo/dbc/gm_global_a_object.dbc",
  "opendbc_repo/dbc/gm_global_a_chassis.dbc",
  "opendbc_repo/dbc/FORD_CADS.dbc",
  "opendbc_repo/dbc/ford_fusion_2018_adas.dbc",
  "opendbc_repo/dbc/ford_lincoln_base_pt.dbc",
  "opendbc_repo/dbc/honda_accord_2018_can_generated.dbc",
  "opendbc_repo/dbc/acura_ilx_2016_can_generated.dbc",
  "opendbc_repo/dbc/acura_rdx_2018_can_generated.dbc",
  "opendbc_repo/dbc/acura_rdx_2020_can_generated.dbc",
  "opendbc_repo/dbc/honda_civic_touring_2016_can_generated.dbc",
  "opendbc_repo/dbc/honda_civic_hatchback_ex_2017_can_generated.dbc",
  "opendbc_repo/dbc/honda_crv_touring_2016_can_generated.dbc",
  "opendbc_repo/dbc/honda_crv_ex_2017_can_generated.dbc",
  "opendbc_repo/dbc/honda_crv_ex_2017_body_generated.dbc",
  "opendbc_repo/dbc/honda_crv_executive_2016_can_generated.dbc",
  "opendbc_repo/dbc/honda_fit_ex_2018_can_generated.dbc",
  "opendbc_repo/dbc/honda_odyssey_exl_2018_generated.dbc",
  "opendbc_repo/dbc/honda_odyssey_extreme_edition_2018_china_can_generated.dbc",
  "opendbc_repo/dbc/honda_insight_ex_2019_can_generated.dbc",
  "opendbc_repo/dbc/acura_ilx_2016_nidec.dbc",
  "opendbc_repo/dbc/honda_civic_ex_2022_can_generated.dbc",
  "opendbc_repo/dbc/hyundai_canfd.dbc",
  "opendbc_repo/dbc/hyundai_kia_generic.dbc",
  "opendbc_repo/dbc/hyundai_kia_mando_front_radar_generated.dbc",
  "opendbc_repo/dbc/mazda_2017.dbc",
  "opendbc_repo/dbc/nissan_x_trail_2017_generated.dbc",
  "opendbc_repo/dbc/nissan_leaf_2018_generated.dbc",
  "opendbc_repo/dbc/subaru_global_2017_generated.dbc",
  "opendbc_repo/dbc/subaru_global_2020_hybrid_generated.dbc",
  "opendbc_repo/dbc/subaru_outback_2015_generated.dbc",
  "opendbc_repo/dbc/subaru_outback_2019_generated.dbc",
  "opendbc_repo/dbc/subaru_forester_2017_generated.dbc",
  "opendbc_repo/dbc/toyota_tnga_k_pt_generated.dbc",
  "opendbc_repo/dbc/toyota_new_mc_pt_generated.dbc",
  "opendbc_repo/dbc/toyota_nodsu_pt_generated.dbc",
  "opendbc_repo/dbc/toyota_adas.dbc",
  "opendbc_repo/dbc/toyota_tss2_adas.dbc",
  "opendbc_repo/dbc/vw_golf_mk4.dbc",
  "opendbc_repo/dbc/vw_mqb_2010.dbc",
  "opendbc_repo/dbc/tesla_can.dbc",
  "opendbc_repo/dbc/tesla_radar_bosch_generated.dbc",
  "opendbc_repo/dbc/tesla_radar_continental_generated.dbc",
  "opendbc_repo/dbc/tesla_powertrain.dbc",
]

# Sunnypilot whitelist
sunnypilot_whitelist = [
  "^README.md",
  ".*selfdrive/test/fuzzy_generation.py",
  ".*selfdrive/test/helpers.py",
  ".*selfdrive/test/__init__.py",
  ".*selfdrive/test/setup_device_ci.sh",
  ".*selfdrive/test/test_time_to_onroad.py",
  ".*selfdrive/test/test_onroad.py",
  ".*system/manager/test/test_manager.py",
  ".*system/manager/test/__init__.py",
  ".*system/qcomgpsd/tests/test_qcomgpsd.py",
  ".*system/updated/casync/tests/test_casync.py",
  ".*system/updated/tests/test_git.py",
  ".*system/updated/tests/test_base.py",
  ".*selfdrive/ui/tests/test_translations.py",
  ".*selfdrive/car/tests/__init__.py",
  ".*selfdrive/car/tests/test_car_interfaces.py",
  ".*selfdrive/navd/tests/test_navd.py",
  ".*selfdrive/navd/tests/test_map_renderer.py",
  ".*selfdrive/boardd/tests/test_boardd_loopback.py",
  ".*INTEGRATION.md",
  ".*HOW-TOS.md",
  ".*CARS.md",
  ".*LIMITATIONS.md",
  ".*CONTRIBUTING.md",
  ".*sunnyhaibin0850_qrcode_paypal.me.png",
  "opendbc/.*.dbc",
]

# Merge the whitelists
whitelist += sunnypilot_whitelist


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
