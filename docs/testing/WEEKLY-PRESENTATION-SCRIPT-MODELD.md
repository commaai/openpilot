# modeld testing progress

## Context

This week we implemented the modeld testing plan in phases, starting with hardware-independent tests and then extending existing integration coverage.
The goal was to improve confidence in modeld correctness while keeping feedback fast in local development.

---

## What we built

We added two new test files and extended one existing file.

First, `test_parse_model_outputs.py` adds pure unit tests for parser math and output transformation behavior, including softmax, sigmoid, missing-output handling, and MDN parsing behavior.

Second, `test_fill_model_msg.py` adds contract tests for how model outputs are mapped into published messages. This verifies core field mapping, list sizes, and constants-driven invariants.

Third, we extended `test_modeld.py` with an additional recovery scenario to verify that model updates recover correctly only after consecutive road frames return.

### JP-Branch
#### Writing New Unit Tests for modeld

##### What we did
- Identified that `selfdrive/modeld/parse_model_outputs.py` and `selfdrive/modeld/constants.py` had **zero unit tests** despite containing critical math and configuration used throughout the driving pipeline.
- Created `selfdrive/modeld/tests/test_parse_model_outputs.py` with 30+ tests covering:
  - `safe_exp` -- clipping behavior, overflow prevention for float16, array support
  - `sigmoid` -- boundary values, output range, monotonicity, symmetry
  - `softmax` -- sum-to-one, non-negativity, numerical stability, different dtypes (float32 vs float16 take different code paths), batch support
  - `Parser.check_missing` -- ValueError on missing keys, ignore_missing flag
  - `Parser.parse_binary_crossentropy` -- sigmoid application, missing key handling
  - `Parser.parse_categorical_crossentropy` -- softmax application, reshaping
  - `Parser.parse_mdn` -- mu/std splitting, positive standard deviations, shape validation, multi-hypothesis path
- Created `selfdrive/modeld/tests/test_constants.py` with 25+ tests covering:
  - `index_function` -- boundary values, quadratic curve, custom parameters
  - `ModelConstants` -- index list lengths/endpoints, scalar constant validation, FCW thresholds
  - `Plan` slices -- correct element counts, non-overlapping, full coverage of PLAN_WIDTH
  - `Meta` slices -- correct element counts for disengage/press/blinker signals

#### Why we chose these files
- **Pure Python + numpy** -- no hardware dependencies, no process management, no camera/VisionIPC. They run instantly on any machine.
- **High impact** -- `parse_model_outputs.py` processes every single neural network output in the driving pipeline. A bug in `sigmoid` or `softmax` would corrupt all model predictions.
- **Zero existing coverage** -- the only existing test (`test_modeld.py`) is an integration test that tests the full modeld process, not these utility functions.

#### What we learned about pytest in this repo

- **`conftest.py` at repo root** has `collect_ignore_glob = ["selfdrive/modeld/*.py"]`. This ignores `.py` files **directly** in `selfdrive/modeld/` (like `modeld.py`, `constants.py`) to prevent them being collected as tests. But it does **NOT** match `selfdrive/modeld/tests/*.py` -- that extra `tests/` directory means our test files are collected normally.
- **Autouse fixtures** run on every test automatically:
  - `openpilot_function_fixture` (function scope) -- sets `random.seed(0)`, creates a clean `OpenpilotPrefix` temp directory, runs `manager.manager_cleanup()` after. This is fine for pure numpy tests.
  - `openpilot_class_fixture` (class scope) -- saves/restores environment variables.
- **`-Werror`** is set in `pyproject.toml` -- any Python warning becomes a test failure. Avoid deprecated APIs.
- **`-n auto`** runs tests in parallel with pytest-xdist. Pure functions with no shared global state are safe.
- **`testpaths`** in `pyproject.toml` includes `"selfdrive"`, so `selfdrive/modeld/tests/` is automatically in scope.

#### How to run just our tests
```bash
pytest selfdrive/modeld/tests/test_parse_model_outputs.py selfdrive/modeld/tests/test_constants.py -v
```
---

## How we validated it

We validated at three levels:

1. Fast unit level:
   - parser tests
   - fill-model-message tests

2. Process integration level:
   - existing modeld process tests plus the new recovery test

3. Coverage comparison level:
   - we added an opt-in script, `scripts/testing/compare_coverage.sh`, that compares baseline/original modeld tests against our new tests only for `selfdrive/modeld`.

This gives us separate baseline and new-test coverage reports without running the entire suite.

Coverage results from this run:

- baseline total: `18.94%`
- new-tests run total: `45.93%`
- net gain: `+26.99` percentage points
- target module improvements:
  - `fill_model_msg.py`: `9.26%` to `96.91%`
  - `parse_model_outputs.py`: `0.00%` to `68.82%`

If someone asks why overall is still not very high: the denominator includes `modeld.py`, `dmonitoringmodeld.py`, and `get_model_metadata.py`, which were not the focus of this week’s scoped run.

---

## Engineering decisions and issues resolved

We kept tests aligned with the existing repository structure by adding tests under `selfdrive/modeld/tests` and reusing root pytest behavior.

During validation we fixed two test-level issues:

- The message test builder mock needed to support both capnp list-init and struct-init patterns.
- A strict float equality assertion was too brittle for float32 values, so we switched to tolerance-based assertions with `pytest.approx`.

These were test harness issues, not modeld production defects.

---

## Outcome and next step

Outcome for this week:

- modeld now has expanded unit, mapping, and integration coverage.
- we can run scoped commands quickly and produce baseline-vs-new coverage artifacts for reporting.
- we now have concrete weekly evidence: total coverage uplift plus module-level gains in parser and message mapping.

Next step:

- use the same layered approach on the next priority subsystem, while keeping coverage comparison as part of weekly evidence.
