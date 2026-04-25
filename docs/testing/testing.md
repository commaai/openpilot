This directory documentation for testing the openpilot project 0.9.8 release done by Group C cluster 1 for the Software Quality and testing course (CS 4223) at the University of Oklahoma under professor Mansoor Abdulhak.

# Table of Contents
* [Assigned Subsystems](#assigned-subsystems)
* [Directories](#directories)
* [Testing tracker](#testing-tracker)
* [Infrastructure overview](#infrastructure-overview)
* [Low-level test plan](#low-level-test-plan)
* [modeld implementation summary](#modeld-implementation-summary)
* [Weekly presentation script](#weekly-presentation-script)

# Assigned Subsystems
* *selfdrive/modeld* (all files)
* *selfdrive/pandad* (assignment-listed files only):
  * *pandad\_api\_impl.pyx*
  * *pandad.cc*
  * *pandad.h*
  * *pandad.py*
  * *SConscript*
  * *spi.cc*
* *system/* (entire directory)

**Pandad — desktop verification aligned with the STP** ([testing-plan/TESTING-PLAN.md](testing-plan/TESTING-PLAN.md) §3.1 unit / §3.3 boundary, risks R2–R3):

* `pytest selfdrive/pandad/tests/test_pandad_can_capnp_*.py selfdrive/pandad/tests/test_pandad_pandad_wrapper.py -q` — Cython CAN serialization (split: roundtrip / event validity / multiblob) and `pandad.py` signature helper (no Panda hardware).
* Native Catch2 USB protocol tests remain in `test_pandad_usbprotocol.cc` (built via SCons); integration and SPI fault-injection stay in `test_pandad_loopback.py` / `test_pandad_spi.py` (`@pytest.mark.tici`).

# Directories

* [testing-plan](testing-plan): Testing plan for the assigned subsystems of openpilot shown in both Markdown and PDF format.

# Testing tracker

* [TESTING-TRACKER.md](TESTING-TRACKER.md): Living backlog of done vs remaining work (modeld gates, pandad, `system/`, infra). Update as you merge or reprioritize.

# Infrastructure overview

* [INFRASTRUCTURE-OVERVIEW.md](INFRASTRUCTURE-OVERVIEW.md): How pytest, root `conftest.py`, and the `selfdrive/` + `system/` support packages fit together; diagrams; where to add tests; fixture cheat sheet.

# Low-level test plan

* [LOW-LEVEL-TEST-PLAN.md](LOW-LEVEL-TEST-PLAN.md): Tactical guide aligned with the STP—repository pytest/native conventions, phased shared infrastructure (`selfdrive/test/support/` and `system/tests/support/`), per-subsystem work breakdown, risk traceability (R1–R10), and scoped commands.

**Modeld Phase C (extra daemon contracts):** `pytest selfdrive/modeld/tests/test_modeld_phase_c_contracts.py -q` — extends §7.1 with subtests; skips if `modeld` never publishes in the environment (anchor `test_modeld.py` unchanged).

**Support harnesses (fixtures + plug-in smoke tests):**

* Selfdrive: `python -m pytest selfdrive/test/support/tests -q` — see `test_selfdrive_support_harness.py`
* System: `python -m pytest system/tests/support/tests -q` — see `test_system_support_harness.py`

If both directories are empty again, their `conftest.py` hooks still map “no tests collected” to exit 0. Fixtures from both `support/fixtures.py` modules load globally via root `pytest_plugins` (`openpilot_params_seeded`, `system_daemon_params`, etc.).

**Modeld coverage comparison (opt-in):**

* Run `scripts/testing/compare_coverage.sh` (executable bit required, or `bash scripts/testing/compare_coverage.sh`). The script passes `scripts/testing/coverage-modeld-compare.ini` (omit **`tests/`**, **`modeld.py`**, **`dmonitoringmodeld.py`**) and runs pytest **`-n 0`** so pytest-cov does not scatter traces across xdist workers (which previously let test modules appear in HTML).
* Default `--ours` in the script matches [LOW-LEVEL-TEST-PLAN.md](LOW-LEVEL-TEST-PLAN.md) §6 plus FCW / `get_model_metadata` helpers; override with `--ours "..."` when needed.
* **`modeld.py`** / **`dmonitoringmodeld.py`** are omitted from this report by design (subprocess entrypoints). **`get_model_metadata.py`** is covered by `test_get_model_metadata_unit.py` (helpers only; the `if __name__ == "__main__"` block is CLI-only).

# modeld implementation summary

* [MODELD-IMPLEMENTATION-SUMMARY.md](MODELD-IMPLEMENTATION-SUMMARY.md): Communication-ready summary of what was implemented for `selfdrive/modeld`, exact run commands, coverage artifacts, and reporting checklist.

# Weekly presentation script

* [WEEKLY-PRESENTATION-SCRIPT-MODELD.md](WEEKLY-PRESENTATION-SCRIPT-MODELD.md): Roughly 3-minute script for a weekly update on modeld testing work, how it was implemented, issues resolved, and next steps.

