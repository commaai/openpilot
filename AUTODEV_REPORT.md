# AUTODEV Report - Issue #37277

## Issue
- #37277 - Replace ModemManager with modem.py
- URL: https://github.com/commaai/openpilot/issues/37277

## Changed files
- `system/hardware/tici/modem.py`
  - Added lightweight modem helper module with:
    - serial AT command execution (`at_cmd`)
    - locked serial context (`ModemPort`) for shared AT access
    - assistance file upload support (`ModemPort.upload_file`)
    - state file helpers for `/dev/shm/modem_state.txt` (`read_modem_state`, `write_modem_state`)
- `system/qcomgpsd/qcomgpsd.py`
  - Replaced `mmcli` AT command usage with `modem.py` helpers.
  - Replaced assistance-data injection through `mmcli` with direct AT file upload path.
  - Updated modem wait logic to use AT helper retries instead of `mmcli` subprocess calls.
- `system/hardware/hardwared.py`
  - Removed ModemManager restart loop in hardware state thread.
- `system/hardware/tici/hardware.py`
  - Added direct AT fallbacks (via `modem.py`) for modem version/IMEI/temperature retrieval when ModemManager DBus is unavailable.
  - Made `get_sim_info` resilient when ModemManager is unavailable.
  - Removed `mmcli` APN clear command from modem configuration path; replaced with direct AT command.
- `system/qcomgpsd/tests/test_qcomgpsd.py`
  - Updated integration test flow to use `lte` service / `lte.sh` actions instead of ModemManager service commands.
  - Removed `mmcli` location-status assertion and replaced with AT-based GNSS state check.
- `system/hardware/tici/restart_modem.sh`
  - Removed ModemManager stop/debug usage; now restarts LTE service after modem power cycle.
- `tests/test_modem_py.py`
  - Added isolated unit tests for the new `modem.py` state-file helpers.

## Validation commands
1. RED (pre-implementation):
   - `uv run --extra testing python -m pytest -q --noconftest -o addopts='' tests/test_modem_py.py`
2. GREEN (post-implementation):
   - `uv run --extra testing python -m pytest -q --noconftest -o addopts='' tests/test_modem_py.py`
3. Lint:
   - `uv run --extra testing ruff check system/hardware/tici/modem.py system/qcomgpsd/qcomgpsd.py system/hardware/hardwared.py system/hardware/tici/hardware.py system/qcomgpsd/tests/test_qcomgpsd.py tests/test_modem_py.py`
4. Syntax compile:
   - `uv run python -m py_compile system/hardware/tici/modem.py system/qcomgpsd/qcomgpsd.py system/hardware/hardwared.py system/hardware/tici/hardware.py system/qcomgpsd/tests/test_qcomgpsd.py tests/test_modem_py.py`
5. Shell syntax:
   - `bash -n system/hardware/tici/restart_modem.sh`

## Validation results
- RED step: failed as expected before implementation (missing `system/hardware/tici/modem.py`).
- GREEN step: `4 passed`.
- Ruff: passed (`All checks passed!`).
- Python compile checks: passed (no output / no errors).
- Shell syntax check: passed (no output / no errors).

## Risks / follow-ups
- `system/hardware/tici/hardware.py` still keeps ModemManager DBus primary path and now uses AT fallbacks; full ModemManager removal across all modem functionality is not yet complete.
- Assistance-data injection now uses direct AT file upload; behavior should be hardware-verified on comma 3X/comma four.
- Integration tests under `system/qcomgpsd/tests/test_qcomgpsd.py` are hardware-specific and were not executed in this environment.
