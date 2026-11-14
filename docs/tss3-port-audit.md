# TSS3 port audit

First audit 2026-09-21; rewritten 2026-10-09 against the current tree (openpilot `cb31aedfe`, opendbc `9bdacdd74`, panda `92eb56516`). Static review, unit tests, and CI-path analysis only — no hardware, live CAN, or road validation, and nothing here certifies actuation safety.

## Verdict

The substantive port is sound and stays: vehicle decoding (`carstate.py`), radar track assembly (`radar_interface.py`), the TSS3 DBCs, independent safety enforcement (`safety/modes/toyota.h`), and standard controller integration. The original audit's worst findings are fixed: panda carries **zero** fork changes (the sample-point and BRS experiments were reverted; the pin is commaai master), the personality-synchronization policy is gone, the redundant radar path is consolidated, and topology is one definition — `ToyotaTSS3PlatformConfig`, toyota_b repin, chassis/state bus 0, FRC bus 2 — covering Camry and Crown. Corolla was deleted wholesale.

Two things are actually open:

1. **Crown has no validation route.** `TOYOTA_CROWN_TSS3` is registered (`values.py:251`) but has neither a route in `tests/routes.py` nor an exemption, so `test_models.py:169` raises `missing test route` and CI (`tests.yml:116`) fails. Beyond that, every TSS3 platform skips the controls tests (`test_models.py:293` — commands are built from live CAN and signer responses), so no TSS3 platform has recorded-route coverage of control generation; Camry's single route (`routes.py:240`) exercises decoding only. `docs/CARS.md` was regenerated for Camry but not Crown.
2. **Bringup automation ships inside the vehicle port.** ~1,650 changed lines across pandad lifecycle changes (`pandad.cc` +330/−13, `tss3_startup_catcher.h`), the oracle daemon and UI (`toyota_tss3_oracle_auto.py`, `tss3_oracle.py`), its tests, and manager registration. It arms on the TSS3 safety bit for any TSS3-flagged platform. The exact-target Corolla signer kit and signed payload now live under `openpilot/selfdrive/car/toyota_tss3_oracle/`, so the git commit fully describes the installed system and no separate `/data/tss3-oracle` deployment is required. It is development tooling, not general vehicle support.

Native startup pre-arm disables automatic CAN-FD promotion on all three Panda buses, matching normal TSS3 safety. This keeps explicitly Classical diagnostic packets Classical after FD wake traffic. The policy is preloaded while OFF, outside the ignition hot path.

Initial ELM327 firmware fingerprinting also disables automatic CAN-FD promotion on all three buses, matching Python Panda diagnostics. On the live Camry, automatic promotion allowed only ABS to answer; disabling it on the same bus 0 restored EPS, camera, and ABS Tester Present/F181 responses. Post-identification frame policy and strict firmware matching are unchanged.

At the 2026-10-09 audit, both forks lagged commaai master. The parent rebase and opendbc pin update are recorded under Validation below; the inventory remains the historical audit snapshot.

## In-repo signer boundary

`tss3.py` signs CONTROL_REQUEST (0x08A) through the **EPS-resident SecOC signer** over CAN — not the external oracle tool. Now 167 lines (down from 331) with a transport-shaped API — `receive` / `request_due` / `send` — and an in-file timing contract: 17–32 ms signer latency hidden by four pending requests, 90 ms response timeout, 100 ms CONTROL_REQUEST gap before the VMC latches a fault. The controller's only coupling is gating rate-limit advance on `request_due` (`carcontroller.py:103`), so driving intent stays in CarController and enforcement stays in safety. Do not merge this into `car/secoc.py`; the upstream helper is keyed authentication with different availability and lifecycle semantics.

Remaining gap: no deterministic offline double drives this boundary, which is why the model suite skips TSS3 controls tests.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Upstream openpilot | Engagement, driver interaction policy, planning, longitudinal/lateral control, personality |
| opendbc Toyota interface/state/radar | Identify supported configuration; decode vehicle observations; expose standard CarParams, CarState, RadarData |
| opendbc Toyota controller | Translate standard CarControl into vehicle commands using existing limiting helpers |
| `tss3.py` SignerTransport | Represent the EPS signer exchange and its availability; no driving policy |
| opendbc safety | Independently restrict actuation and forwarding, with bounded failure behavior |
| panda | Unmodified upstream transport |
| Bringup (to be split out) | Prepare and inspect the development environment through a separately maintained lifecycle |

## Remaining work

1. Register a Crown route or exempt it deliberately — CI is red on `test_models` until then. Recover capture provenance: the deleted audit fixtures' `manifest.json` was the only record, and the Camry route's comment says only "openpilot longitudinal". Build a deterministic signer double and un-skip the TSS3 controls tests against it. Regenerate `CARS.md`.
2. Split bringup into its own review series: pandad startup/lifecycle, oracle daemon + UI, and external tool versioning. Until then it travels with every port change and vice versa.
3. Parent update completed 2026-10-13 onto `42ccc5ceb`; opendbc now uses the updated `tss3-expand` branch at `9de3b22d`.

## Validation

Run 2026-10-09 with `opendbc_repo/.venv`:

- opendbc Toyota suites (`car/toyota/tests/`, `safety/tests/test_toyota.py`): **318 passed, 124 skipped, 272 subtests passed**. Skips are pre-existing platform/SecOC skips plus the TSS3 controls skip above.
- Parent TSS3 suites (`pandad/tests/test_direct_panda_lease.py`, `car/tests/test_toyota_tss3_oracle_*.py`): **19 passed, 22 subtests passed**.
- Previously recorded (2026-09): five recorded-route checks passed against the Camry capture; Toyota safety suites pass with their existing skips.
- Not run: full CI, MISRA coverage, hardware builds, live CAN, road tests.

### 2026-10-13 rebase verification

- Replayed the parent port onto commaai master `42ccc5ceb`, preserving both upstream navigation settings and the TSS3 controls in the mici developer page.
- Pinned opendbc `tss3-expand` at `9de3b22d`, including the upstream car-mass removal and the Camry, Crown, and Corolla TSS3 definitions.
- Rebuilt `openpilot/common/libparams_c.dylib` for the updated parameter keys.
- Parent oracle suites: **16 tests passed** under unittest. Panda lease suite: **3 standalone test functions passed** under `unittest.FunctionTestCase`.
- Desktop smoke: rendered and visually checked navigation and TSS3 bringup controls; confirmed TSS3 eligibility for Camry, Crown, and Corolla. The external signer kit was absent, and the UI displayed `KIT MISSING`.
- LFS downloads were skipped during rebase; this does not fix Hugging Face authentication. Full builds, Toyota safety suites, hardware, and road behavior were not validated in this update.

### 2026-10-21 rebase test repair

- Lease lifecycle tests now use `unittest.TestCase` so `tools/test_runner.py` discovers all three scenarios.

### 2026-10-22 CONTROL_REQUEST checksum and PCS correction

- `CONTROL_REQUEST` byte 7 is the Toyota checksum over the first eight bytes. CarState and panda detect PCS from `PCS_BRAKE_REQUEST` (byte 4, mask `0x40`); the checksum-derived request IDs are removed. Panda validates the checksum before approving signer requests in both longitudinal modes.
- Replayed `00000043--29caa20fbc--12` through CarInterface and compiled panda safety: all 21 recorded PCS frames were detected and forwarded after RX, with normal control restored afterward. All 1,662,572 readable incoming `CONTROL_REQUEST` frames in the retained logs passed the checksum; LogReader reported corrupted events in the corpus.
- Toyota car/safety and CAN suites: **370 passed, 40 skipped** under unittest. MISRA, Ruff, codespell, and C lint passed. All **22 mutations** of the changed Toyota safety logic were killed.

### 2026-11-01 Corolla clean-baseline stationary capture

- Capture `20261101T201200-1690-35111683423` was produced by a clean checkout
  of openpilot `80b8615c39`, opendbc `d8a50afdbe`, and upstream panda
  `f38f504ffa`; `initData.dirty` was false. The signed Panda firmware on the
  device matched the checkout.
- The external kit matched Corolla EPS F181
  `8965F1208000 / 8A3111213000`, installed payload
  `6234bb5bd496a82fabb32c27b56b3db0f1907b5bf6482e4c19f8835153ff69fc`,
  and received one status-zero `0x7A9` response to its four-frame `0x777`
  probe in 68.691 ms. All four installer transmissions were accepted and none
  were rejected.
- The route stayed stationary and CAN-valid for 39.265 seconds, but exact
  firmware matching returned `MOCK`. Panda therefore remained in ELM327 safety
  and runtime signer probing never started; this capture does not test signer
  residency past eight seconds or the Toyota safety transition.
- The car reported engine `8A2221221000`, ABS `F152612D1700`, EPS
  `8965F1208000 / 8A3111213000`, radar `8821F4801200`, and camera
  `8646F210200`. The engine, ABS, and radar variants are now added to the
  Corolla exact-match set alongside the existing EPS entry; the camera's
  nonstandard 11-character response can remain an extra ECU
  because exact matching succeeds without making it a fuzzy-match authority.
- No host `0x08A` was published. The mirrored native `0x08A` carried equal
  upper/lower OEM acceleration fields from -0.178 to -0.169 m/s². The
  persisted `AlphaLongitudinalEnabled` parameter was `1`; it must be disabled
  before the next stationary run so the first recognized Corolla test remains
  in stock-longitudinal mode.

### 2026-11-01 Corolla recognized-route topology correction

- Three later stationary segments fingerprinted as `TOYOTA_COROLLA_TSS3` with
  dashcam mode disabled, stock longitudinal selected, and vehicle speed zero.
  After Panda changed from ELM327 to Toyota safety, the Corolla's full-rate
  vehicle stream remained on physical bus 2. The existing TSS3 configuration
  expected control-critical state on bus 0, so its RX checks became invalid and
  the route reported `controlsMismatch`, `canBusMissing`, `canError`, and
  `signerUnavailable`.
- Corolla now sets a dedicated safety flag that validates its TSS3 RX checks on
  bus 2, and CarState reads its powertrain parser from bus 2. Camry and Crown
  retain their bus-0 topology. Signer requests (`0x777`) and host
  `CONTROL_REQUEST` (`0x08A`) remain outputs on bus 0; the native FRC `0x08A`
  remains an input on bus 2.
- The captured blind-spot frame `0x3F6` was present only during startup and
  disappeared after the safety transition. Corolla therefore no longer sets
  `HAS_BSM`, preventing this optional frame from invalidating otherwise healthy
  vehicle state.
- Replaying the captured traffic through the corrected CarInterface produced
  1,100/1,100 CAN-valid updates over an 11-second window in segment 3 and
  3,000/3,000 CAN-valid updates over a 30-second window in segment 5, with no
  CAN timeout in either window. The Toyota car suite passed 102 tests with 2
  skips; the focused TSS3 safety suites passed 56 tests with 4 skips.
- Segments 3 and 4 each contained one four-frame runtime `0x777` request and a
  status-zero `0x7A9` response, but CAN invalidation stopped continued probing.
  No host `0x08A` was published. `AlphaLongitudinalEnabled` was absent and
  `openpilotLongitudinalControl` remained false. These captures therefore do
  not yet establish continuous signer authority beyond eight seconds or
  authorize moving-road validation; the corrected build needs another
  stationary route.

### 2026-11-02 automatic startup handoff correction

- Bundle `20261102T104036-1732-77870428342` identifies the exact deployed
  sources as openpilot `e0cf509f4d`, opendbc `53c7c0b92d`, and upstream Panda
  `f38f504ffa`; the signed Panda firmware matched the checkout.
- Two manual installs bound the exact Corolla EPS F181
  `8965F1208000 / 8A3111213000`, installed payload
  `6234bb5bd496a82fabb32c27b56b3db0f1907b5bf6482e4c19f8835153ff69fc`,
  and received status-zero `0x7A9` signer responses in 67.696 ms and
  74.165 ms. Each probe had four accepted and zero rejected publications.
- The automatic path sent `10 02` only 0.253 ms after the exact positive
  `50 03`, but native pandad immediately released Panda ownership. The warm
  installer resumed about 9.6 ms after the marker reached its daemon and saw
  application F181 rather than the boot endpoint. In both successful manual
  captures the native startup loop first observed the exact
  `7F 10 78 00 00 00 00` programming-pending response about 15 ms after
  `10 02`; this was the missing transition proof in the automatic handoff.
- Native pandad now retains Panda ownership after sending `10 02` and publishes
  the existing v1 handoff marker only after receiving that exact pending
  response on bus 0. The marker adds response frame and monotonic timing fields,
  while retaining the kit-compatible schema and verdict. A missing or different
  response times out without handing an unconfirmed transition to the uploader.
- This bundle contains installer telemetry, not qlog/rlog. It therefore does
  not revalidate route fingerprinting, Toyota safety CAN validity, continuous
  runtime signer authority, host `0x08A`, or longitudinal fields. Another
  stationary route remains required after the corrected automatic install.

### 2026-11-02 Corolla safety-transition signer topology

- Bundle `20261102T161420-1668-38125473056` ran the exact clean parent
  `ed20ddd26d`, opendbc `53c7c0b92d`, and Panda `f38f504ffa`. The included
  installer run was the manual startup-race path, not the native automatic
  handoff, so it does not exercise the response-gated handoff above. It did
  install the expected payload and received a status-zero `0x7A9` response in
  75.163 ms with four accepted and zero rejected transmissions.
- Both routes fingerprinted exactly as `TOYOTA_COROLLA_TSS3`, remained in Park
  at 0 m/s, selected stock longitudinal, and used Toyota safety parameter
  `12873` (`TSS3_RX_BUS_2` included). Segment 6 had 4,625/4,625 CAN-valid
  CarState updates; segment 7 had 5,487/5,537 valid updates, with the 50 startup
  invalid samples occurring before the required bus-2 messages arrived. Panda's
  Toyota RX checks remained valid. There were no `canError` or `canBusMissing`
  events and no stable-Toyota-safety `controlsMismatch`.
- The first runtime four-frame `0x777` probe was sent while Panda still used
  ELM327 safety. It appeared on both bus 0 and bus 2 and received status-zero
  `0x7A9` at 10.52 ms. At the Toyota safety transition the relay opened; later
  bus-0 probes continued to receive Panda TX receipts but no longer appeared on
  bus 2 and received no responses. Across the rest of segment 7 and segment 6,
  100 complete probes were unanswered with zero Toyota-safety TX rejections.
  This locates the failure at the signer request egress topology, not signer
  residency: the payload was still live more than eight seconds after install
  immediately before the relay transition.
- Corolla now sends only signer `0x777` fragments on bus 2 and parses their
  rejected loopback on bus 194. Signed host `0x08A` remains on bus 0 toward the
  VMC, with its independent rejection loopback on bus 192. Panda safety permits
  bus-2 signer fragments only when the Corolla topology flag is set and rejects
  bus-0 signer fragments in that mode; Camry and Crown remain on bus 0.
- No host `0x08A` was published in either stationary route. Both CarControl and
  CarOutput acceleration stayed exactly 0.0, `openpilotLongitudinalControl` was
  false, and `AlphaLongitudinalEnabled` was absent. Moving-road validation is
  still not authorized until a new stationary route proves continuous bus-2
  signer responses and host publication behavior.

### 2026-11-02 Corolla bus-2 signer validation

- Bundle `20261102T174008-1686-55426954509` ran the exact clean parent
  `f52a872b18`, opendbc `e59da8d8db`, and Panda `f38f504ffa`. The installed
  Panda reported `DEV-f38f504f-DEBUG`; its signature prefix `5423f5e1fdf937be`
  matched the rebuilt firmware file. The kit bound the exact Corolla EPS F181,
  installed payload
  `6234bb5bd496a82fabb32c27b56b3db0f1907b5bf6482e4c19f8835153ff69fc`,
  and received a status-zero live-probe response in 65.439 ms with four
  accepted and zero rejected transmissions. No persistent flash write, payload
  upload, or ECU reset was used.
- The newly routed bus-2 runtime path remained live under Toyota safety. One
  run received 55 valid status-zero `0x7A9` responses in uninterrupted signer
  sequence 1 through 55 over 54.001 seconds; 54 of those responses followed
  the Toyota-safety transition. Its 55 four-frame `0x777` requests produced
  all 220 accepted bus-2 receipts and no bus-194 rejection receipt. A separate
  steady-state run continued uninterrupted sequence 56 through 82 for another
  25.990 seconds, with all 108 fragments accepted and none rejected. This
  closes the prior relay-egress failure and establishes signer residency well
  beyond eight seconds.
- Firmware matching was exact and non-fuzzy as `TOYOTA_COROLLA_TSS3`, with
  dashcam mode disabled and Toyota safety parameter `12873`. In the startup
  run, the first 40 CarState samples were invalid only while required frames
  populated; the following 5,510 samples were CAN-valid. The steady-state run
  was CAN-valid for all 2,684 samples. Panda's Toyota RX checks stayed valid,
  and neither run reported `canError` or `canBusMissing`. The one startup
  `signerUnavailable` event cleared on the first valid response. The lone
  `controlsMismatch` sample occurred only when the other run deliberately left
  Toyota safety for ELM327 installer mode; it was not present in the stable
  Toyota interval.
- Both runs remained in Park at 0 m/s with stock longitudinal selected.
  CarControl and CarOutput acceleration were exactly 0.0 throughout. Every
  logged `0x08A` was the 32-byte native FRC request or its byte-identical Panda
  passthrough; no host-signed `0x08A` was published because controls were never
  engaged. The single aggregate `safetyTxBlocked` increment occurred in the
  ELM327 startup transition and reset to zero on entry to Toyota safety; there
  were no rejected signer or host-publication loopbacks. The stationary test
  therefore validates fingerprinting, CAN/safety health, continuous signer
  transport, and zero longitudinal output, but an engaged stationary capture
  is still required to validate payload-matched host `0x08A` publication before
  moving-road validation.

### 2026-11-02 Corolla engagement-attempt capture

- Bundle `20261102T191831-1682-32848698684` ran the exact clean documented
  parent `ff74fd1dff`, opendbc `e59da8d8db`, and Panda `f38f504ffa`. The Panda
  signature prefix `5423f5e1fdf937be` again matched the rebuilt firmware. The
  kit bound the exact Corolla EPS F181, installed the expected
  `6234bb5bd496a82fabb32c27b56b3db0f1907b5bf6482e4c19f8835153ff69fc`
  payload, and received a status-zero live-probe response in 66.522 ms with all
  four transmissions accepted and none rejected.
- Runtime fingerprinting remained exact and non-fuzzy as
  `TOYOTA_COROLLA_TSS3`, stock longitudinal remained selected, and Toyota
  safety parameter `12873` was active. After the 53 startup samples populated
  the required frames, 5,486 consecutive CarState samples were CAN-valid.
  Panda's Toyota RX checks remained valid, with no `canError`, `canBusMissing`,
  or `controlsMismatch` event.
- The bus-2 signer path again produced 55 uninterrupted status-zero `0x7A9`
  responses in sequence 1 through 55 over 54.009 seconds. All 220 fragments of
  the corresponding 55 `0x777` requests had accepted bus-2 receipts, with no
  bus-194 rejection receipt. The single aggregate `safetyTxBlocked` increment
  occurred during the ELM327-to-Toyota startup transition and reset to zero in
  Toyota safety.
- This was not the missing engaged stationary test. The vehicle shifted through
  Park, Reverse, Neutral, and briefly Drive, reaching 1.402 m/s. The decoded
  cruise-main state remained zero, all decoded MAIN/SET/RES/CANCEL button bits
  remained zero, and no cruise button event was recorded. Consequently
  SelfdriveState remained disabled, CarControl was never enabled or laterally
  active, and `CRUISE_OPERATING_LATCH` remained zero.
- No host `0x08A` was sent. All 2,439 native 32-byte FRC requests matched either
  their pre-relay bus-0 copy or their post-relay Panda passthrough, and there
  was no bus-192 host-publication rejection. CarControl and CarOutput
  acceleration remained exactly 0.0, but host payload matching is still
  untested. This capture requires no code change: a new test must keep the car
  stopped in Drive with the service brake held, release the parking brake, turn
  cruise main on, and issue the normal SET/RES engagement input while retaining
  stock longitudinal.

### 2026-11-03 Corolla repeated unengaged capture

- Archive `qlog-11.zip` ran the exact clean documented parent `ab872bbab5`,
  opendbc `e59da8d8db`, and Panda `f38f504ffa`. Its two installer captures both
  matched Panda signature prefix `5423f5e1fdf937be`, installed the expected
  signer payload, and received status-zero live-probe responses in 68.668 ms
  and 74.157 ms. Each probe had four accepted and zero rejected transmissions.
- Runtime fingerprinting was again exact and non-fuzzy as
  `TOYOTA_COROLLA_TSS3`, with stock longitudinal and Toyota safety parameter
  `12873`. After 75 startup samples populated the required frames, all 5,765
  remaining CarState samples were CAN-valid. There was no `canError`,
  `canBusMissing`, or `controlsMismatch` event, and Panda's Toyota RX checks
  remained valid.
- The signer returned 57 uninterrupted status-zero bus-2 `0x7A9` responses in
  sequence 1 through 57 over 56.013 seconds. All 228 fragments of the matching
  57 `0x777` requests had accepted bus-2 receipts and none had a bus-194
  rejection receipt. Bus 2 reported a historical total-error count of 427 in
  its first Panda sample, but that value never rose; receive/transmit error
  counters, warning/passive state, and bus-off all remained zero.
- This capture also did not engage controls and was not stationary. It reached
  5.442 m/s while shifting from Park through Reverse to Drive. The cruise-main
  display state remained zero, the decoded MAIN/SET/RES/CANCEL bits remained
  unchanged at their released values, no button event was emitted, and the
  stock `CRUISE_OPERATING_LATCH` remained zero. SelfdriveState therefore stayed
  disabled and CarControl was never enabled or laterally active.
- No host `0x08A` was sent or rejected. All 2,442 native FRC requests matched
  their pre-relay physical bus-0 copy or post-relay Panda passthrough. The stock
  request's two longitudinal fields varied together from -0.570 to 1.341 m/s²
  while the vehicle moved, but both openpilot CarControl and CarOutput
  acceleration remained exactly 0.0. Host-signed payload and zero-longitudinal
  field validation therefore remain pending, and this archive supports no
  functional code change.

### 2026-11-03 native handoff and engaged-runtime captures

- Bundle `20261103T123619-1667-42448363212` ran the exact clean parent
  `8b0fbfa2ba`, opendbc `e59da8d8db`, and Panda `f38f504ffa`. Panda's device
  and rebuilt-file signature prefixes both matched `5423f5e1fdf937be`; the
  installed payload SHA-256 was
  `6234bb5bd496a82fabb32c27b56b3db0f1907b5bf6482e4c19f8835153ff69fc`.
- The native automatic catcher is now proven end to end. It observed exact
  `50 03` 240.534 ms after ignition, sent `10 02` 0.261 ms later, received
  exact pending `7F 10 78` 20.304 ms after that, and notified the daemon in
  time for backend launch 0.623 ms after receipt. The warm installer then
  received a status-zero signer response in 8.264 ms with all four request
  fragments accepted and none rejected.
- The backend nevertheless reported failure because Brake did not support the
  optional `19 02 FF` DTC query. All supported control-domain observations
  contained no explicit fault, and this Brake query has been unavailable in
  the retained Corolla captures. The kit now treats this peer result as an
  explicit warning, runs the independent fresh signer self-test, and fails only
  an observed peer fault or failed signer test (kit source `f612bdc`). The
  vehicle-side daemon also preserves that warning instead of replacing it with
  a generic success line.
- Both runtime routes fingerprinted exactly and non-fuzzily as
  `TOYOTA_COROLLA_TSS3`, used Toyota safety parameter `12873`, kept Panda RX
  checks valid, and recorded no `canError`, `canBusMissing`, or
  `controlsMismatch`. MAIN, SET, RES, and CANCEL changed their expected decoded
  bits. Controls engaged laterally for 1,886 samples in segment 13 and 966 in
  segment 14; `longActive` and CarControl/CarOutput acceleration remained zero.
- Segment 13 produced 1,909 status-zero bus-2 `0x7A9` responses and 1,869 host
  `0x08A` publications. Segment 14 produced 1,009 such responses and 962 host
  publications. Every host payload in both routes exactly equaled a preceding
  28-byte `0x777` application plus that request's fresh four-byte response
  trailer, with 2.226--27.049 ms response-to-publication latency. Panda accepted
  all 2,831 host publications and rejected none.
- Each route had one rejected final `0x777` fragment. In both cases the first
  three fragments were accepted while controls were enabled, then a pedal or
  steering override disabled controls before the final fragment's receipt.
  Panda rejected that in-flight fragment and the transport released ownership;
  no host publication was rejected. This is the intended disengagement safety
  boundary, not a stable-runtime signer rejection.
- These are not stationary validation routes: peak speeds were 13.053 and
  11.744 m/s. With stock longitudinal selected, the host request deliberately
  copied the FRC request's equal upper/lower longitudinal fields, so they were
  often nonzero while stock ACC was active. Zero openpilot longitudinal output
  is proven, but the requested stationary zero-field check is not. Zeroing the
  copied fields in code would violate the stock-longitudinal safety contract;
  an actually stationary engaged capture is still required for that criterion.

### 2026-11-04 post-repin stationary capture: chassis still on bus 1

- Bundle `rlog-18.zip` contains two stationary segments from clean openpilot
  commit `936f67ffd6cae0452b3f3b1750fa7b85963e800d` on
  `star2025-kai-clean`. Both exact-match attempts returned `MOCK`, so Panda
  remained in ELM327 diagnostic safety and the UI correctly kept oracle
  bringup in its disabled `CHECK HARNESS` state.
- The physical topology did not change to the required Toyota-B repin. Across
  both segments, chassis/PT IDs appeared only on bus 1: `0x025` had 11,662
  frames, `0x0AA` 11,656, `0x101` 5,827, `0x081` 3,885, and native `0x08A`
  4,661. Buses 0 and 2 instead carried the same bridged CAN-FD camera/FRC
  family while the interception relay remained closed. This is electrically
  the unrepinned arrangement, not a transient fingerprint or UI failure.
- No signer request `0x777` was present. Ten bus-1 `0x7A9` frames in segment
  18 had no matching signer request and occurred during diagnostic discovery;
  they are not signer-response evidence. There was no host `0x08A`
  publication, no controls-allowed sample, and no Toyota safety transition.
- All 11,141 CarControl samples kept `enabled`, `latActive`, and `longActive`
  false, with actuator acceleration, gas, and brake exactly zero. The only
  onroad events were `carUnrecognized`, startup initialization, and
  `startupNoCar`; there were no `canError`, `canBusMissing`, or
  `controlsMismatch` events, but that cannot validate Toyota safety while the
  car is `MOCK`.
- Do not route around this state in software and do not road-test it. With
  power removed, correct and retain the complete Toyota-B pair swap so the
  chassis side is CAN0/bus 0 and the camera/FRC side is CAN2/bus 2, then repeat
  stationary fingerprint and signer validation.

### 2026-11-13 repeat harness capture: topology unchanged

- `rlog (1).zst` came from the clean deployed documentation commit
  `34f6e8518fc4e85086d5efed5c18d82eb9997ea2` on
  `star2025-kai-clean`. Its active 60-second window again fingerprinted as
  `MOCK`, emitted 60 `carUnrecognized` events, and kept Panda in ELM327 safety
  for all 600 reported samples.
- Bus placement is unchanged. Bus 1 carried 81,576 frames, including all
  chassis/PT traffic: 6,003 `0x025`, 6,000 `0x0AA`, 3,000 `0x101`, 2,000
  `0x081`, and 2,399 native `0x08A`. Buses 0 and 2 each carried exactly 23,639
  frames of the same relay-bridged CAN-FD family. The intended car-side CAN0
  and camera-side CAN2 interception topology is therefore still not present.
- There were no `sendcan` frames, signer `0x777` requests, `0x7A9` responses,
  host `0x08A` publications, controls-allowed samples, or Toyota-safety
  samples. All 6,000 CarControl messages kept `enabled`, `latActive`, and
  `longActive` false, with acceleration, gas, and brake exactly zero.
- This repeat rules out a one-off UI or fingerprint transient. No software
  routing change is appropriate; power-off continuity and terminal-retention
  checks must prove the complete pair exchange before another stationary run.

### 2026-11-14 repin recovery capture: functional bus-2 topology restored

- Archive `qlog.zip` contains two sequential stationary routes beginning at
  03:09:47 and 03:10:22 UTC from clean openpilot commit `34f6e8518f` on
  `star2025-kai-clean`, plus bringup bundle
  `20261113T221013-1710-65860944588`. Both routes recovered the exact,
  non-fuzzy `TOYOTA_COROLLA_TSS3` firmware fingerprint, selected stock
  longitudinal, and used Toyota safety parameter `12873`.
- The first route preceded the successful RAM install. Its 19 complete
  bus-2 `0x777` requests produced 76 accepted and zero rejected fragment
  receipts, but no valid runtime signer response; `signerUnavailable`
  consequently remained present. The included kit then bound Corolla EPS
  F181 `8965F1208000 / 8A3111213000`, matched the Panda device and signed-file
  signature prefix `5423f5e1fdf937be`, and retained payload SHA-256
  `6234bb5bd496a82fabb32c27b56b3db0f1907b5bf6482e4c19f8835153ff69fc`.
  Its independent self-test received a status-zero `0x7A9` response in
  66.724 ms with all four `0x777` fragments accepted and none rejected. The
  peer-state verdict remained explicitly inconclusive because the optional
  Brake DTC query timed out; no supported peer-health check reported a fault.
- The second, post-install route proves stable runtime residency. It completed
  58 bus-2 signer requests, received all 58 status-zero bus-2 `0x7A9`
  responses, and recorded all 232 request fragments as accepted with no
  bus-194 rejection. The only `signerUnavailable` event was the startup sample
  before the first response. After 39 startup samples populated the required
  frames, the remaining 5,790 CarState samples were CAN-valid. Panda reported
  no fault and no invalid Toyota RX check; neither route emitted `canError`,
  `canBusMissing`, or `controlsMismatch`.
- The observed side assignment is the known bus-2-adapted arrangement. In the
  post-install route, the full-rate `0x025`, `0x0AA`, `0x101`, and `0x081`
  stream arrived on physical bus 2 after the relay transition and was
  forwarded to bus 0. This is why the existing Corolla-specific bus-2 parser,
  safety RX flag, and signer routing operate correctly in this capture. It
  records the observed topology without treating another in-car repin as a
  safe or necessary validation step.
- Both routes stayed in Park at exactly 0 m/s. Every CarControl and CarOutput
  acceleration sample was 0.0, and CarControl gas, brake, `enabled`,
  `latActive`, and `longActive` also remained zero or false. Controls never
  engaged, so there was no host `sendcan` `0x08A` to payload-match. All 2,439
  post-install native FRC `0x08A` frames were accounted for by 149 pre-relay
  bus-0 copies and 2,290 accepted bus-0 passthrough receipts; none is evidence
  of a host publication. Host-publication rejection and host longitudinal
  fields therefore remain untested.
- Preserve this now-functional harness and do not revert the bus-2 software
  adaptation. The preceding terminal move produced the documented bus-1/MOCK
  failure, so no further terminal swap should be attempted from software bus
  observations alone. The next test is an engaged stationary capture on this
  exact configuration for payload-matched host `0x08A`, zero host-publication
  rejection, and both host longitudinal fields at zero. If an upstream-style
  CAN0/CAN2 physical layout is pursued later, first map the disconnected
  harness end to end with power-off continuity measurements and an unambiguous
  cavity-numbered pinout; do not trial that change in the car.

### 2026-11-14 engaged stock-longitudinal and alpha diagnostic routes

- The additional route files all ran clean deployed openpilot commit
  `34f6e8518f` on `star2025-kai-clean` with exact, non-fuzzy Corolla
  fingerprinting. They are moving-road captures, not the requested stationary
  validation: peak speed was 15.437 m/s. Panda reported no fault or invalid RX
  check. The single `controlsMismatch` event occurred at the end of `rlog-4`
  on the deliberate Toyota-to-ELM327 safety transition, not during stable
  Toyota safety.
- The stock-longitudinal `rlog.zst` interval later in the 03:10:22 UTC route
  engaged lateral control for 2,056 samples. It produced 2,071 complete signer
  requests, 2,072 status-zero responses (including one response whose request
  preceded the retained interval), and 2,057 accepted host `0x08A`
  publications with none rejected. The retained interval payload-matched
  2,055 publications; the other two began across the omitted segment boundary.
  Openpilot longitudinal control and acceleration remained zero. The equal
  host longitudinal fields ranged from -0.214 to -0.004 m/s² because stock
  mode correctly copied the moving vehicle's native longitudinal request.
- `rlog-1.zst` and `rlog-2.zst` are one continuous alpha-longitudinal route.
  The user deliberately enabled alpha after perceiving no openpilot output;
  it was not a spontaneous parameter or controller transition. CarParams
  consequently reported `openpilotLongitudinalControl=true`, Toyota safety
  parameter `12361`, and 5,650 `longActive` CarControl samples. Openpilot
  acceleration reached 1.780 m/s², and the equal host longitudinal fields
  ranged from -0.454 to 1.780 m/s². All 5,709 host publications exactly
  matched their retained 28-byte request and fresh response trailer, all were
  accepted, and none was rejected. Four signer-fragment rejections occurred
  across disengagement boundaries; no host publication was rejected.
- `rlog-3.zst` and `rlog-4.zst` then returned to stock longitudinal and provide
  direct evidence that openpilot was producing lateral output. CarControl was
  enabled and `latActive` for 6,510 samples while `longActive`, CarControl
  acceleration, and CarOutput acceleration remained zero. The transport made
  6,541 complete signer requests, received 6,527 status-zero responses, and
  published 6,491 host `0x08A` frames. Every host payload exactly matched its
  application and fresh trailer, Panda accepted all 6,491, and rejected none;
  response-to-publication latency was 2.231--26.358 ms.
- All 6,491 stock-mode host publications requested lateral result ID 11, and
  1,438 retained `CONTROL_RESULT` frames reported lateral result ID 11 without
  `REQUEST_LOSS`. That result is not proof that the host request reached the
  actuator: the native bus-2 request also used lateral ID 11 in 1,724 frames,
  so the result cannot be attributed uniquely to the host publication.
  CarOutput commanded up to 99.993° and the measured/requested angles had
  0.912 correlation, but the rate limiter itself follows measured angle. When
  the host target differed from measured angle by more than 5°, measured
  angle moved toward it after 250 ms in only 55.2% of samples. This proves
  openpilot generation and Panda transmission, not steering authority.
- The user's report of no felt actuation therefore keeps the relay-side
  hypothesis open. Runtime signer requests and EPS responses are on bus 2,
  while the completed host `0x08A` is transmitted on bus 0. If those are
  opposite sides of the opened interception relay in this harness, Panda can
  accept every correctly signed publication without the VMC applying it. This
  is the specific topology failure previously predicted by the external port
  author; alpha longitudinal cannot correct it because it changes payload
  fields, not the publication bus.
- Do not tune angle limits or gains and do not make a production routing
  change from these moving captures. Leave alpha off. The next diagnostic must
  be stationary and must distinguish bus-0 from bus-2 host delivery while
  retaining safety enforcement; only an observed actuator response may select
  a publication side. The requested stationary zero-host-longitudinal-field
  criterion remains outstanding.

### 2026-11-14 parked follow-up: no control engagement

- Archive `qlog-3.zip`, subsequently supplied as four qlog/rlog pairs and
  bringup bundle `20261113T225652-1685-39687139932`, ran clean deployed
  documentation commit `29d73dc50f`, opendbc `e59da8d8db`, and Panda
  `f38f504ffa`. Both automatic catches matched the signed Panda firmware,
  retained the expected signer payload, accepted all four independent probe
  fragments with none rejected, and passed fresh status-zero self-tests in
  68.985 and 69.304 ms. Neither install changed host publication routing.
- Both stationary routes fingerprinted exactly as `TOYOTA_COROLLA_TSS3`, used
  stock longitudinal and Toyota safety parameter `12873`, kept Panda RX checks
  valid, and reported no Panda fault, `canError`, or `canBusMissing`. Runtime
  probing remained continuous across the four retained segments: 213 complete
  bus-2 requests produced 852 accepted fragments, zero rejected fragments,
  and 213 status-zero responses. The startup `signerUnavailable` sample in
  each route cleared normally. The one `controlsMismatch` sample coincided
  with the final Toyota-to-ELM327 transition.
- This was not a host-delivery or steering-authority diagnostic. All 21,442
  CarControl samples had `enabled=false`, `latActive=false`, and
  `longActive=false`; every Panda sample had `controlsAllowed=false`. The car
  remained in Park at 0 m/s with parking-brake, seatbelt, wrong-gear, and
  wrong-car-mode no-entry conditions. No host `sendcan` `0x08A` was generated
  on any bus, so there is no bus-0/bus-2 comparison or actuator-response
  evidence in this archive.
- CarControl and CarOutput acceleration, gas, and brake remained zero. The
  large steering-angle values in the final segment came from manually moving
  the wheel while controls were disabled; the controller output reference
  followed measured angle and was limited to 99.993°. Those values are not
  openpilot steering commands. No functional code or harness change follows
  from this capture. A valid next diagnostic must actually engage laterally
  while stationary and explicitly vary only the host-publication side under
  safety enforcement.

### 2026-11-14 two-route safety restore failure

- The 24 qlog/rlog pairs supplied with bringup bundle
  `20261114T124935-1690-42163918159` contain two moving routes from clean
  deployed commit `b4269f6c3d`. The bundle matched Panda's device and signed
  firmware signatures, retained payload SHA-256
  `6234bb5bd496a82fabb32c27b56b3db0f1907b5bf6482e4c19f8835153ff69fc`,
  and passed both installer and independent status-zero signer probes with
  four accepted and zero rejected fragments. It did not contain the proposed
  bus-2 host-publication diagnostic; completed host `0x08A` routing remained
  on bus 0.
- The first route began at 17:49:43 UTC, fingerprinted exactly as
  `TOYOTA_COROLLA_TSS3`, selected stock longitudinal, and entered Toyota
  safety parameter `12873`. It engaged lateral control in multiple retained
  segments and published thousands of accepted signed host `0x08A` frames on
  bus 0. CarControl `longActive`, CarControl acceleration, and CarOutput
  acceleration remained zero throughout. This is further evidence of host
  generation under the existing routing, not proof of actuator authority.
- At the end of that route the automatic startup catcher armed for the next
  ignition and deliberately selected ELM327 safety parameter 1. The second
  route began at 18:11:15 UTC. Its exact Corolla CarParams again requested
  Toyota safety parameter `12873`, but every retained PandaState instead
  remained ELM327 parameter 1. It logged a `10 02` request 0.251 ms after the
  exact positive `50 03`, then timed out before confirming the PROGRAMMING
  transition. No subsequent `setting safety model: 2, param: 12873` message
  appeared.
- Consequently the second route continuously reported `controlsMismatch`
  and `signerUnavailable`. It reached 21.78 m/s, but every CarControl sample
  kept `enabled`, `latActive`, and `longActive` false, Panda never allowed
  controls, and there were no runtime status-zero `0x7A9` responses or host
  `0x08A` publications. This directly explains why openpilot did not command
  the car on that drive; it is independent of the unresolved bus-0/bus-2 host
  delivery question.
- Root cause is a stale `PandaSafety::safety_configured_` cache. Pandad used to
  skip `configureSafetyMode(false)` whenever the startup catcher was active.
  The catcher therefore changed the hardware to ELM327 offroad without
  invalidating the previous route's cached Toyota configuration. When the
  next catch timed out, `configureSafetyMode(true)` incorrectly concluded
  that safety was already configured and left Panda in ELM327.
- Pandad now always runs the non-transmitting offroad cache reset even while
  the startup catcher owns the hardware policy. It still does not overwrite
  an active catch onroad. Thus a timed-out catch causes the next onroad safety
  pass to install the current CarParams Toyota policy. A four-state regression
  test covers active/inactive catcher behavior onroad and offroad.

## File inventory (merge-base diffs vs commaai master, 2026-10-09)

### opendbc — 22 files, +2,258 / −57

| File | +/- | Disposition |
| --- | --- | --- |
| `docs/CARS.md` | +2 / −1 | Regenerate; missing Crown. |
| `opendbc/can/dbc.py` | +4 / −1 | Keep; checksum dispatch for TSS3. |
| `opendbc/car/tests/routes.py` | +1 / −0 | Keep; add Crown route and provenance comments. |
| `opendbc/car/tests/test_fw_fingerprint.py` | +2 / −2 | Keep; query-time budget updates. |
| `opendbc/car/tests/test_models.py` | +2 / −0 | Replace blanket TSS3 skip with signer double. |
| `opendbc/car/torque_data/override.toml` | +2 / −0 | Keep. |
| `opendbc/car/toyota/carcontroller.py` | +59 / −1 | Keep; TSS3 update path. |
| `opendbc/car/toyota/carstate.py` | +168 / −2 | Keep; TSS3 decoding. |
| `opendbc/car/toyota/fingerprints.py` | +22 / −0 | Keep; captured identities. |
| `opendbc/car/toyota/interface.py` | +28 / −4 | Keep; TSS3 params wiring. |
| `opendbc/car/toyota/radar_interface.py` | +81 / −36 | Keep; consolidated TSS3 path. |
| `opendbc/car/toyota/tests/test_toyota.py` | +562 / −4 | Keep; merged TSS3 CarInterface, signer, and radar tests, parameterized over TSS3 platforms. |
| `opendbc/car/toyota/toyotacan.py` | +92 / −2 | Keep; message construction, checksums cover up to the CHECKSUM byte. |
| `opendbc/car/toyota/tss3.py` | +167 / −0 | Keep; SignerTransport. |
| `opendbc/car/toyota/values.py` | +62 / −3 | Keep; Camry + Crown definitions. |
| `opendbc/dbc/generator/toyota/toyota_tss3_pt.dbc` | +189 / −0 | Keep; byte-7 checksums and CONTROL_REQUEST PCS_BRAKE_REQUEST decoding. |
| `opendbc/dbc/generator/toyota/toyota_tss3_radar.py` | +67 / −0 | Keep. |
| `opendbc/safety/modes/toyota.h` | +317 / −5 | Keep; independent TSS3 enforcement. |
| `opendbc/safety/tests/test_toyota.py` | +475 / −1 | Keep; merged TSS3 safety tests. |

### openpilot — 24 files, +1,926 / −65

| File | +/- | Disposition |
| --- | --- | --- |
| `.gitmodules` | +2 / −1 | Keep while fork pins are needed. |
| `opendbc_repo`, `panda` | pins | Keep. |
| `openpilot/common/params_keys.h` | +1 / −0 | Bringup series. |
| `openpilot/selfdrive/car/tests/test_toyota_tss3_oracle_kit.py` | +65 / −0 | Bringup series. |
| `openpilot/selfdrive/car/tests/test_toyota_tss3_oracle_reporting.py` | +171 / −0 | Bringup series. |
| `openpilot/selfdrive/car/toyota_tss3_oracle_auto.py` | +363 / −0 | Bringup series. |
| `openpilot/selfdrive/car/toyota_tss3_oracle_kit.py` | +30 / −0 | Bringup series. |
| `openpilot/selfdrive/car/toyota_tss3_oracle_status.py` | +58 / −0 | Bringup series. |
| `openpilot/selfdrive/pandad/main.cc` | +1 / −2 | Bringup series. |
| `openpilot/selfdrive/pandad/panda.cc` | +15 / −0 | Split: generic explicit-FD protocol vs bringup helper. |
| `openpilot/selfdrive/pandad/panda.h` | +2 / −1 | Split: protocol vs bringup. |
| `openpilot/selfdrive/pandad/panda_safety.cc` | +12 / −0 | Keep; scoped TSS3 safety param. |
| `openpilot/selfdrive/pandad/pandad.cc` | +330 / −13 | Bringup series. |
| `openpilot/selfdrive/pandad/pandad.h` | +3 / −1 | Bringup series. |
| `openpilot/selfdrive/pandad/pandad.py` | +143 / −27 | Bringup series. |
| `openpilot/selfdrive/pandad/tss3_startup_catcher.h` | +8 / −0 | Bringup series. |
| `openpilot/selfdrive/pandad/tests/test_direct_panda_lease.py` | +231 / −0 | Bringup series. |
| `openpilot/selfdrive/pandad/tests/test_pandad_canprotocol.cc` | +7 / −0 | Keep; protocol test. |
| `openpilot/selfdrive/ui/mici/layouts/main.py` | +3 / −1 | Bringup series. |
| `openpilot/selfdrive/ui/mici/layouts/settings/developer.py` | +38 / −2 | Bringup series. |
| `openpilot/selfdrive/ui/mici/layouts/settings/tss3_oracle.py` | +206 / −0 | Bringup series. |
| `openpilot/system/manager/process_config.py` | +4 / −0 | Bringup series. |
| `uv.lock` | +2 / −16 | Keep; consequence of pins. |

### panda — no fork changes

The pin equals commaai master `92eb56516`. Nothing to review or carry.
