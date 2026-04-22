# Low-level test plan and execution guide

This document turns the [Software Test Plan (STP)](testing-plan/TESTING-PLAN.md) into actionable work: where tests live, how they hook into existing tooling, what to build first, and how to keep traceability and quality bar consistent with the upstream openpilot codebase.

**Audience:** implementers writing or extending tests for `selfdrive/modeld`, the assignment-scoped `selfdrive/pandad` files, and `system/`.

---

## 1. Ground truth: how this repo runs tests

### 1.1 Pytest configuration

- **Config:** `[tool.pytest.ini_options]` in `pyproject.toml` defines `testpaths`, markers (`slow`, `tici`, `skip_tici_setup`), strict markers/config, warnings as errors, and `xdist` defaults (`-n auto`, `loadgroup`).
- **Global fixtures and environment:** Root `conftest.py` applies an autouse function fixture that:
  - seeds `random` for determinism;
  - wraps each test in `OpenpilotPrefix` (isolated prefix / download cache behavior);
  - restores environment and calls `manager.manager_cleanup()` after each test;
  - re-enables GC if a test disabled it.
- **Hardware:** Tests marked `@pytest.mark.tici` are skipped automatically on non-TICI hosts and receive `tici_setup_fixture` on device.

**Implication:** New tests must assume this lifecycle. Avoid persisting global process state across tests; use `managed_processes[...].start()` / `.stop()` or helpers that guarantee teardown.

### 1.2 Existing shared helpers (reuse before adding new layers)

- `selfdrive/test/helpers.py` — `set_params_enabled`, `processes_context` / `with_processes`, HTTP test server utilities, `release_only` decorator.
- Subsystem-specific bases — e.g. `system/updated/tests/test_base.py` for updater scenarios; follow the same pattern for new subsystems when setup is non-trivial.

### 1.3 Native / C++ tests

- Pandad USB protocol tests live as compiled gtests (e.g. `selfdrive/pandad/tests/test_pandad_usbprotocol.cc`); build wiring is via SCons in-tree.
- Keep **Python integration** and **C++ unit** boundaries clear: protocol math in C++; process and messaging behavior in pytest where the rest of the stack does.

---

## 2. Design principles (professional patterns)

1. **Arrange–Act–Assert** in each test; prefer small tests over scenario soup.
2. **One primary concern per test** (or parameterized table for pure logic).
3. **Determinism:** fixed seeds, explicit timeouts, no wall-clock assumptions without tolerance; use existing `random.seed(0)` behavior from global fixture.
4. **Isolation:** params and processes reset via fixtures/teardown; do not rely on execution order across test modules.
5. **Traceability:** every new test maps to at least one STP risk (R1–R10) or explicit requirement ID in a docstring or comment block at class level.
6. **Markers:** use `@pytest.mark.slow` for expensive tests; `@pytest.mark.tici` only when hardware is required; do not skip desktop CI silently without a marker strategy.
7. **Contracts at boundaries:** favor tests that assert **message fields, ordering, and side effects** at IPC/API edges (black-box) and reserve white-box tests for fault injection and specific branches (as with SPI fault injection today).

---

## 3. Concrete testing infrastructure (phased)

The repository already *is* the primary infrastructure (pytest + `conftest.py` + `selfdrive/test/helpers.py`). The gap is **team-owned structure** for your course scope. Adopt the following **without forking upstream patterns unnecessarily**.

### Phase 0 — Conventions (no code, same week as first tests)

| Artifact | Purpose |
|----------|---------|
| **Risk tag in test docstrings** | First line of class or function: `Maps: R3, R6` matching STP risks. |
| **Subsystem prefixes** | Keep tests under existing trees: `.../tests/test_*.py` next to the code under test. |
| **Scoped pytest commands** | Document exact commands per folder in section 6 below for CI parity locally. |

### Phase 1 — Shared package (implemented)

**Selfdrive** — layout under `selfdrive/test/support/`:

| Module | Role |
|--------|------|
| `processes.py` | `managed_process_scope(...)` — context manager over `processes_context` |
| `params_seed.py` | `seed_minimal_openpilot_params()` — same baseline as `set_params_enabled` |
| `messaging.py` | `new_live_calibration_message(...)` and room for more builders |
| `fixtures.py` | Pytest fixtures: `openpilot_params_seeded`, `managed_processes_ctx`, `pub_sub_factory` |

**System** — layout under `system/tests/support/` (inside the existing `system/tests/` tree already listed in `testpaths`; reuses selfdrive `managed_process_scope`):

| Module | Role |
|--------|------|
| `processes.py` | Re-exports `managed_process_scope` from `openpilot.selfdrive.test.support.processes` |
| `params_seed.py` | `seed_system_daemon_params()` (e.g. `IsOffroad`, `DongleId`); `seed_full_stack_params()` (+ minimal selfdrive seed) |
| `messaging.py` | `make_pub_sub(pub, sub, **kw)` for `PubMaster` / `SubMaster` pairs |
| `fixtures.py` | Pytest fixtures (prefixed `system_`): `system_daemon_params`, `system_full_stack_params`, `system_managed_processes_ctx`, `system_pub_sub_factory` |

Root `conftest.py` registers both plugins so any test under `testpaths` can request fixtures by name:

`openpilot.selfdrive.test.support.fixtures` and `openpilot.system.tests.support.fixtures`.

**Harness directories:** each has a `tests/` folder with `conftest.py` mapping `ExitCode.NO_TESTS_COLLECTED` to `OK` if the suite is ever empty again. Smoke tests live in `test_selfdrive_support_harness.py` and `test_system_support_harness.py`.

```bash
python -m pytest selfdrive/test/support/tests -q
python -m pytest system/tests/support/tests -q
```

**Rule:** Keep subsystem-specific setup (e.g. VisionIPC for modeld, loggerd segment layout) next to those tests; move repeated building blocks into the appropriate `support/` package after the second copy.

### Phase 2 — Fixtures (pytest)

- Prefer **function-scoped** fixtures for processes and sockets.
- Use **class-scoped** fixtures only when setup is expensive *and* `openpilot_class_fixture` + env rules in `conftest.py` are respected (see comment in root `conftest.py` about `setUpClass` and env).
- For `xdist`, use `@pytest.mark.xdist_group(...)` when tests share a non-isolated resource (pattern already registered in `pytest_configure`).

### Phase 3 — Data and assets

| Data type | Approach |
|-----------|----------|
| Synthetic modeld frames | Reuse pattern: fixed-size zero buffer + `VisionIpcServer` as in `test_modeld.py`; consider committing a **minimal** binary fixture only if a regression is tied to pixel layout. |
| CAN / params | Build in-test via `cereal` and `Params().put(...)`; avoid mystery blobs. |
| Replay segments | Use upstream replay tooling when validating integration; keep unit tests independent of large downloads where possible. |

---

## 4. Work breakdown by subsystem (low-level)

### 4.1 `selfdrive/modeld`

**Existing anchor:** `selfdrive/modeld/tests/test_modeld.py` — VisionIPC + pub/sub + `managed_processes['modeld']`; asserts frame IDs, timestamps, drop behavior.

| Priority | Type | Focus | Maps to |
|----------|------|--------|---------|
| P0 | Integration | Keep E2E path green (frame progression, dropped-frame policy) | R1, correctness row in STP table |
| P1 | Unit | Pure-Python helpers (e.g. `parse_model_outputs` / parser utilities if in scope for the assignment) | Maintainability, R1 |
| P2 | Nonfunctional | Timing thresholds (document p95 or max latency in test name and assertion message) | Performance row |

**Implementation notes:**

- Mirror `setup_method` / `teardown_method` lifecycle for any new process-based test.
- When adding cases, extend `_send_frames` / `_wait` or extract to `selfdrive/test/support/` only after duplication.

### 4.2 `selfdrive/pandad` (assignment files only)

**Existing anchors:** `test_pandad.py` (hardware), `test_pandad_loopback.py`, `test_pandad_spi.py`, `test_pandad_usbprotocol.cc`.

| Priority | Type | Focus | Maps to |
|----------|------|--------|---------|
| P0 | C++/gtest | Protocol packing/unpacking, buffer edge cases | R2, R3 |
| P0 | Integration | Loopback / transport integrity | R3 |
| P1 | White-box | SPI fault injection paths in `spi.cc` | R3 |
| P1 | System/HW | Firmware recovery (marked `tici`) | R2 |

**Implementation notes:**

- Python tests: follow loopback/SPI patterns (env vars for fault injection where designed in C++).
- C++ tests: add cases next to existing gtest files; run via the same SCons/native target the tree already uses.

### 4.3 `system/`

**Breadth strategy:** STP already maps subfolders to tests (`manager`, `loggerd`, `camerad`, `athena`, `webrtc`, `updated`, `hardware`, etc.).

| Priority | Type | Focus | Maps to |
|----------|------|--------|---------|
| P0 | Integration | Manager process graph lifecycle | R4 |
| P0 | Integration | Logger record/encode/delete | R5 |
| P1 | Integration / mocked | Athena / webrtc session and failure modes | R6 |
| P1 | Device / stress | Camera timing, power (often `tici` or slow) | R7, R10 |
| P2 | Integration | Update staging integrity | R8 |
| P2 | Device | GPS/sensor suites | R9 |

**Implementation notes:**

- Prefer extending existing `tests/` packages under each component.
- Use `selfdrive/test/helpers.py::processes_context` or `openpilot.system.tests.support.managed_process_scope` when multiple managed processes must run together.
- For network-facing code, follow `http_server_context` / handler patterns already in `helpers.py`.

---

## 5. Traceability matrix (STP risks to evidence)

Use this when opening PRs or writing the test section of tickets.

| Risk | Meaning (short) | Primary existing evidence | Typical new work |
|------|------------------|---------------------------|------------------|
| R1 | modeld stale/mismatched outputs | `test_modeld.py` | More edge cases; parser unit tests if in scope |
| R2 | Pandad safety mode / heartbeat | `test_pandad.py`, loopback | Assertions on safety-related messaging where observable |
| R3 | Transport / Cython boundary | SPI, USB gtest, loopback | Targeted fault and boundary tests |
| R4 | Manager orchestration | `system/manager/test/test_manager.py` | Restart/kill scenarios if allowed by env |
| R5 | Logger integrity | `system/loggerd/tests/*` | Corruption / backpressure cases |
| R6 | Athena / webrtc misuse | `system/athena/tests/*`, `system/webrtc/tests/*` | Additional mock server cases |
| R7 | Camera instability | `system/camerad/test/*` | Timing regression tests (may be device-only) |
| R8 | Update integrity | `system/updated/tests/*`, `selfdrive/test/test_updated.py` | Overlay / staging invariants |
| R9 | GPS/sensor inconsistency | ublox/sensord/qcomgpsd tests | Parser fixtures from traces |
| R10 | Thermal/power | `system/hardware/tici/tests/*` | Policy threshold tests on device |

---

## 6. Commands (local and scoped)

Run from repository root with the same `PYTHONPATH` / venv the project expects (see CI docker invocations in `.github/workflows/` for parity).

**Examples:**

```bash
# Full Python suite (heavy; matches broad CI intent)
pytest

# Shared support harnesses (empty until you add tests; exit 0)
pytest selfdrive/test/support/tests
pytest system/tests/support/tests

# Modeld only
pytest selfdrive/modeld/tests/

# Modeld phased rollout (fast -> integration)
pytest selfdrive/modeld/tests/test_parse_model_outputs.py -q
pytest selfdrive/modeld/tests/test_fill_model_msg.py -q
pytest selfdrive/modeld/tests/test_modeld.py -q

# Coverage comparison: baseline/original vs new modeld tests (opt-in)
bash scripts/testing/compare_coverage.sh \
  --cov-target selfdrive/modeld \
  --baseline "selfdrive/modeld/tests/test_modeld.py" \
  --ours "selfdrive/modeld/tests/test_parse_model_outputs.py selfdrive/modeld/tests/test_fill_model_msg.py"

# Pandad Python tests (prefer explicit files; test_pandad.py is device-heavy / tici)
pytest selfdrive/pandad/tests/test_pandad_loopback.py
pytest selfdrive/pandad/tests/test_pandad_spi.py
pytest selfdrive/pandad/tests/test_pandad.py   # on TICI or when marked tests are not skipped

# Manager
pytest system/manager/test/

# Loggerd
pytest system/loggerd/tests/

# Skip slow tests
pytest -m "not slow"

# Single test debugging
pytest path/to/test_file.py::TestClass::test_method -vv --tb=short
```

Native gtests are invoked via the build system after `scons` (or the component’s documented target); keep Makefile/SCons targets aligned with how Jenkinsfile builds the relevant artifacts.

---

## 7. Definition of done (per test or PR)

- [ ] Test fails on an intentional bug injection (or was validated to fail before fix).
- [ ] No new warnings; `PYTHONWARNINGS=error` passes locally when used.
- [ ] Teardown leaves no stray processes (rely on `openpilot_function_fixture` + explicit stops).
- [ ] STP risk IDs (`R*`) noted in docstring or PR description.
- [ ] Appropriate markers: `slow`, `tici`, etc.
- [ ] If a new pattern is duplicated three times, extract to `selfdrive/test/support/` or `system/tests/support/` (by layer) with a one-module responsibility.

### 7.1 modeld rollout gates

- [ ] Phase A parser tests pass standalone (`test_parse_model_outputs.py`) on WSL without starting managed daemons.
- [ ] Phase B message-mapping tests pass standalone (`test_fill_model_msg.py`) and validate `ModelConstants`-driven array lengths.
- [ ] Phase C daemon contract tests pass (`test_modeld.py`) with no new flakiness.
- [ ] Coverage comparison script outputs both reports (`baseline`, `ours`) under `.coverage-compare/modeld/`.
- [ ] Combined modeld suite passes before merge: `pytest selfdrive/modeld/tests -q`.

---

## 8. Relationship to other docs

- [testing.md](testing.md) — scope and index for course documentation.
- [TESTING-PLAN.md](testing-plan/TESTING-PLAN.md) — strategic STP (risks, V&V, lifecycle).
- This file — **tactical** execution: layout, reuse, phases, and subsystem checklists.

Update this low-level plan when you add new shared modules or change pytest markers/paths so desktop vs device expectations stay explicit.
