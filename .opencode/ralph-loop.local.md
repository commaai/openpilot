---
active: false
iteration: 2
maxIterations: 100
---

Remove git submodules and move them to pyproject.toml dependencies using git URLs pointing to master branch.

COMPLETED:
1. ✅ Added 6 git-based dependencies to pyproject.toml:
   - msgq @ git+https://github.com/commaai/msgq.git@master
   - opendbc @ git+https://github.com/commaai/opendbc.git@master
   - pandacan @ git+https://github.com/commaai/panda.git@master
   - rednose @ git+https://github.com/commaai/rednose.git@master
   - teleoprtc @ git+https://github.com/commaai/teleoprtc.git@master
   - tinygrad @ git+https://github.com/tinygrad/tinygrad.git@master

2. ✅ Removed submodule references from tool configs:
   - pytest: removed --ignore flags for submodules
   - codespell: removed submodule paths from skip list
   - ruff: removed submodule paths from exclude list
   - ty: removed submodule paths from exclude list

3. ✅ Updated scripts to remove submodule management:
   - tools/op.sh: removed submodule checking and update commands
   - selfdrive/test/setup_device_ci.sh: removed submodule commands
   - release/build_stripped.sh: removed submodule deinit and cleanup
   - release/build_release.sh: removed submodule checks
   - release/check-submodules.sh: deleted file
   - selfdrive/test/test_updated.py: removed submodule setup
   - .github/workflows/repo-maintenance.yaml: removed submodule update step
   - selfdrive/ui/installer/installer.cc: removed --recurse-submodules and submodule update

4. ✅ Deleted .gitmodules file

5. ✅ Removed submodule directories and symlinks:
   - msgq_repo/, opendbc_repo/, panda/, rednose_repo/, teleoprtc_repo/, tinygrad_repo/
   - Symlinks: msgq, opendbc, rednose, teleoprtc, tinygrad
