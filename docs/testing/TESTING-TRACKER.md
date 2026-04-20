# Testing tracker (living document)

**Purpose:** Single place to record what testing work is **done**, **in progress**, and **not started** for the course scope (`selfdrive/modeld`, listed `selfdrive/pandad` files, `system/`). Update this file when you merge tests, open tickets, or change priorities.

**Related plans (strategy and tactics):**

- [Software Test Plan (STP)](testing-plan/TESTING-PLAN.md) — risks R1–R10, scope, verification and validation approach.
- [Low-level test plan](LOW-LEVEL-TEST-PLAN.md) — pytest layout, phases, commands, definition of done.

**Conventions:**

- Use `Maps: R#` in new test docstrings (see LOW-LEVEL §Phase 0).
- Mark items `[x]` when merged and stable in `main` (or your integration branch).
- Mark items `[~]` for in-progress (optional; or use a short note under the item).

---

## Summary

| Subsystem | Done (high level) | Open / next |
|-----------|-------------------|-------------|
| modeld | Parser unit suite; fill + integration test files exist | Extend fill/integration coverage; optional timing tests |
| pandad | Upstream anchors (gtest, loopback, SPI, device tests) | Team-owned extra cases per STP R2/R3 |
| system | Upstream tests per component | Team-owned extensions per LOW §4.3 P0–P2 |
| Infra | Shared `support/` packages + pytest plugins | Optional first tests in support harness dirs |

---

## modeld (`selfdrive/modeld/tests/`)

Aligned with [LOW-LEVEL §7.1](LOW-LEVEL-TEST-PLAN.md#71-modeld-rollout-gates) rollout gates.

| Status | Item | Location / command |
|--------|------|---------------------|
| [x] | **Phase A:** `Parser`, `safe_exp`, `sigmoid`, `softmax`, MDN/crossentropy paths; Ruff clean | `selfdrive/modeld/tests/test_parse_model_outputs.py` |
| [ ] | **Phase B:** Expand `ModelConstants`-driven shape and field coverage for `fill_model_msg` (optional keys, edge batch sizes) | `selfdrive/modeld/tests/test_fill_model_msg.py` |
| [ ] | **Phase C:** Additional R1 daemon contract cases (frame/timestamp/drop policy, camera combos) | `selfdrive/modeld/tests/test_modeld.py` |
| [ ] | **Phase C (optional):** Documented latency / timeliness threshold in test name + assertion | `selfdrive/modeld/tests/test_modeld.py` or new file in same dir |
| [ ] | **Coverage compare (opt-in):** Run `scripts/testing/compare_coverage.sh` and archive reports under `.coverage-compare/modeld/` | See [testing.md](testing.md) snippet |
| [ ] | **Gate:** Full suite green before merge | `pytest selfdrive/modeld/tests -q` |

**Existing files (not necessarily “complete” for course goals):**

- `test_constants.py` — constants / contract checks as implemented.
- `test_fill_model_msg.py` — message fill tests (extend as needed).
- `test_modeld.py` — VisionIPC + managed `modeld` integration.

---

## pandad (assignment-scoped files)

| Status | Item | Location |
|--------|------|----------|
| [ ] | Extra USB protocol / buffer edge cases | `selfdrive/pandad/tests/test_pandad_usbprotocol.cc` |
| [ ] | Additional loopback / transport integrity | `selfdrive/pandad/tests/test_pandad_loopback.py` |
| [ ] | SPI fault-injection / retry coverage | `selfdrive/pandad/tests/test_pandad_spi.py` |
| [ ] | Device-heavy recovery / safety-adjacent (`tici` as required) | `selfdrive/pandad/tests/test_pandad.py` |

**Maps:** R2, R3 (and R2 for device safety flows).

---

## `system/`

Priorities from [LOW-LEVEL §4.3](LOW-LEVEL-TEST-PLAN.md#43-system).

| Priority | Status | Item | Location |
|----------|--------|------|----------|
| P0 | [ ] | Manager lifecycle / graph (restart-kill-reconnect if env allows) | `system/manager/test/` |
| P0 | [ ] | Logger encode / delete / backpressure or corruption scenarios | `system/loggerd/tests/` |
| P1 | [ ] | Athena session / auth / failure (mocked) | `system/athena/tests/` |
| P1 | [ ] | WebRTC session / failure modes | `system/webrtc/tests/` |
| P1 | [ ] | Camera timing regression (`slow` / `tici` as needed) | `system/camerad/test/` |
| P2 | [ ] | Update staging / overlay integrity | `system/updated/tests/`, `selfdrive/test/test_updated.py` |
| P2 | [ ] | GPS / sensor parser fixtures | `system/ubloxd/tests/`, `system/sensord/tests/`, `system/qcomgpsd/tests/` |
| P2 | [ ] | Thermal / power policy (device) | `system/hardware/tici/tests/` |

**Maps:** R4–R10 per subsystem (see LOW-LEVEL traceability matrix).

---

## Shared testing infrastructure

| Status | Item | Location |
|--------|------|----------|
| [x] | Selfdrive support: processes, params seed, messaging builders, pytest fixtures | `selfdrive/test/support/` |
| [x] | System support: re-exports + system fixtures | `system/tests/support/` |
| [x] | Root `pytest_plugins` registration | Root `conftest.py` |
| [ ] | First real tests in harness dirs (optional; today may collect 0 tests) | `selfdrive/test/support/tests/`, `system/tests/support/tests/` |
| [ ] | Extract duplicated setup to support after **third** copy | Per LOW-LEVEL §7 |

---

## Nonfunctional themes (STP §7.2)

Track as **additional cases** in the rows above, not as orphan workstreams.

| Status | Theme | Typical target |
|--------|--------|----------------|
| [ ] | Stress / soak with manager-controlled processes | `system/manager/test/` |
| [ ] | Disk pressure / uploader retry behavior | Loggerd + related upload tests |
| [ ] | Security / privacy failure behavior | `system/athena/tests/`, `system/webrtc/tests/` |

---

## Changelog (optional)

Edit when you want a paper trail without git archaeology:

| Date | Change |
|------|--------|
| 2026-04-20 | Initial tracker; Phase A modeld parser suite marked done. |
