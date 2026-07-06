# Style review v2 — post-cleanup verification (2026-07-05)

Verification pass over the v1 review cleanup. Every claim below was re-measured against the
tree or re-tested in the GUI harness, not taken from PROGRESS.md. Calibration targets remain
`system/loggerd/` and `system/camerad/`.

**Verdict: the cleanup is real.** The tree has genuinely moved to system/ style — dead
abstractions deleted, test-manufactured API gone, contracts intact, and the ratchet is wired
into `run_smoke.sh` so it stays that way. What remains is a short, mostly mechanical punch
list (§2) plus two functional items that were claimed-adjacent but not actually fixed (§3).

---

## 1. Verified fixed (measured, not claimed)

| v1 item | Evidence now |
|---|---|
| A1 dead observer system | 0 `Event<>` in tree (was 7 members / 16 emit sites) |
| A2 PaneRegistry | gone; static pane table with plain function pointers |
| A4 test-manufactured API | 12 of 13 pane headers are a single `draw_*_pane()`; panes_smoke: 133 helper calls → 5 |
| A5 micro-modules | `undo.{h,cc}` folded into dbc; `native_dialog.cc` 270→152 (kept as a file — acceptable, it is a thing) |
| A6 config/result ceremony | `PlaybackClockConfig`/`PlaybackAdvanceResult` gone |
| A8 runtime.cc god-file | 1,719 → 825 lines; remote-routes/autosave/live-source split out |
| B1 error out-params | 37 → **0** `std::string *error` in headers |
| B2 null-guard writes | 59 → **0** |
| B3 Session accessors | 26 getters → public members; `DBCManager` now owned by Session |
| B4 per-frame JSON | `PaneInstance::transient_state` (typed, with a correct constraint comment); JSON only at load/save |
| B6 globals/statics | `dbc()` singleton gone; pane-local statics **0** (was growing) |
| B7 InputText dance | `input_text_with_hint(label, hint, std::string*)` via `imgui_stdlib` |
| B9 comments | 0 → house density on the threaded code (store 6/510, ingest 11/210, live 11/840) — constraint comments, not narration |
| B10 frame-path allocs | `Store::seriesPathsMatching(filter, limit)`; popup no longer copies 10,976 paths/frame |
| Perf appendix #1 | 3-camera playback: **5.69 ms CPU settled** (was 28.4 ms) — under the 8 ms gate, verified live |
| Ratchet | `tests/style_ratchet.sh` exists and runs in `run_smoke.sh` |

---

## 2. Remaining punch list (style)

### 2.1 Naming rename — DONE (correction 2026-07-05, second look)

The v2 claim that this was skipped was a measurement error: the grep glob matched
`generated_event_extractors.h` (capnp calls), polluting the count. The backend rename to
snake_case (`assign_sources`, `add_signal`, `close_all`, reference error params) landed in the
final implementer batch. One fallout bug was found and fixed during baseline verification: the
sweep had renamed **cereal capnp API calls** in `tests/live_smoke.cc` (`setEnabled` →
`set_enabled`) and typo'd `poller.start_(` — the build was red until repaired. Remaining check
for the style pass: confirm no other capnp/ImGui/external API call got swept (grep for
`set_[a-z_]+\(` / `init_[a-z_]+\(` against capnp builder objects), and that the ratchet's
camel check stays 0.

### 2.2 Layering inversion — DONE (verified at Phase-1 baseline)

`panes/messages.h` is now 9 lines; the message-id/CSV utilities live in `backend/csv.{h,cc}`;
the `export.h` shim is deleted. (`kLoggySeriesPathPayload` in browser.h and
`map_basemap_effective_cache_root` in map.h remain as the two accepted small exceptions.)

### 2.3 Header structs — AUDITED, floor is 73 (done 2026-07-05 evening)

Full audit executed against the boundary rule. Result: 79 → **73**, with a per-struct
disposition table in the audit commits. The earlier ≤40 target was wrong: the survivors
genuinely cross boundaries (Store's 11 are its concurrency/query API; transport/workspace
types are backend↔shell↔panes seams; every video/panda struct has ≥2 real callers in the
camera pane). Also removed: 10 dead perf-metric fields, a zero-caller accessor
(`priority_order()`, test rewritten to the real boundary), `RouteResolveConfig` merged away,
`ComputedPythonSpec` flattened. Ratchet baseline now 73; may drift a little lower if Phase-3
moves pane logic to the backend, but do not force it below the boundary rule's honest floor.

### 2.4 Pimpl came back (v1 A3): 3 instances

`backend/video.h` (×2) and `backend/panda_live.h`. Defensible motive (keeps FFmpeg/libusb out
of every include), but note `loggerd/video_writer.h` includes FFmpeg directly — the house
answer. Either inline the members, or keep pimpl **only** for the ffmpeg/libusb pair and say so
in a one-line comment. Don't let it spread past these two.

### 2.5 LOC — SETTLED at the measured floor: 20,595 (ceiling frozen 20,605)

Phase-3 outcome (2026-07-05 late): three independent read-only analyses converged on ~680
lines of genuine zero-capability-loss cuts — not the hoped-for 3,200 — because the earlier
de-abstraction passes already took the real fat and the standing rules protect the repetition
that remains. All ~680 were executed (live/route extraction dedup, dead metadata pipeline,
dead exports, shell_quote/settings consolidation, plot serializer collapse, forward-decl
ceremony) plus integrator review fixes. 21,269 → **20,595**. Owner accepted the measured
floor over the aspirational 18k ("let's do that cleanup and move on"); do not squeeze
further — every remaining line either delivers capability or is protected house style.
Historical target notes below kept for context:

### (superseded) LOC: 21,241 product lines — revised target: ≤ 18,000 (owner-approved 2026-07-05)

The original ≤15k target predates the full feature surface (workspace engine, live-source
matrix, computed-series editor). Adeeb has signed off on **18k** as the final product-LOC
budget; it supersedes the plan's §2.14 number. Current split: panes 9,173 / backend 6,140 /
dbc 1,658 / shell 4,270 / main 239 (tests excluded).

Route there (~3.2k to cut, no capability loss): §2.2 layering fix, §2.3 single-caller struct
inlining, per-pane boilerplate dedup in the worst lines-per-feature panes, and tightening
`backend/live.cc` (840) and `backend/route.cc` (659). Do NOT squeeze past 18k at the cost of
readability or features — that violates "don't overcorrect". Ratchet accordingly: product LOC
baseline steps down with each punch-list commit and freezes at 18,000.

### 2.6 HUD p99 — DONE (Phase 1)

Now a fixed-size ring buffer aged by wall clock (5 s window, no per-frame allocation). QA
verified: spike to 8.0 ms on a triple-seek storm decays back to ~5.2 ms within 10 s.

---

## 3. Functional items — status after Phase 1 (2026-07-05 evening)

1. ~~Two-click selection~~ **DONE** — fixed in the final implementer batch; verified by
   adversarial QA across 6+ single-click transitions incl. during playback and with filters.
2. **`tools/cabana/panda.cc` is still modified** (reference tree, +2/−1). The fix itself is
   fine; it needs Adeeb's explicit sign-off or a revert + loggy-side workaround. STILL OPEN.
3. ~~Preset pane composition~~ **DONE** — cabana preset: road camera top-right over
   binary+plot; jotpluggler preset: camera preview under browser, and after QA feedback the
   plot is now the hero (~70% of window, fractions 0.22/0.78 root split).
4. **Fixed from QA findings (Phase 1):** wheel-scroll false row highlight in Messages
   (ImGui stale Selectable fill; now re-asserted from owned state per frame); clipped Message
   editor at 1280×720 (bounded scrollable child).
5. ~~Silent idle crash~~ **EXPLAINED — not a product bug** (2026-07-05 late). Reproduced under
   controlled soak: process exits with **status 0** (clean), empty log, healthy RSS — i.e. the
   7D signal handler doing its job on a stray external SIGTERM. This box runs many concurrent
   agents whose cleanup kills by process name; every historical "silent crash" (original
   review, QA idle death, soak death at ~4–12 min) coincided with concurrent agent activity,
   and a 12-min gdb session with no concurrent kills survived. Hardening landed: the runtime
   now logs `loggy: exiting on signal N` on signal-initiated shutdown, so any future death
   self-identifies. Falsifier: a death WITHOUT that stderr line reopens this as a real bug.
6. **Coverage note:** `tests/panes_smoke.cc` is now an 81-line boundary stub (by design after
   v1-A4), but PROGRESS.md still describes the old 668-assertion suite — update PROGRESS and
   confirm the moved-to-backend logic (find tools, history comparators) kept its tests.

### Harness lesson (bake into every QA brief)
`xdotool mousemove X Y click 1` as ONE command is silently dropped by headless GLFW (~always):
motion+press+release arrive within one frame, before the app polls the new cursor position —
this produced a false "tab switching broken" blocker. Always two-step:
`xdotool mousemove X Y; sleep 0.3; xdotool click 1`. Hover styling can look identical to
selected styling in screenshots — verify by content change, not highlight.

---

## 4. Ratchet — new baselines (lock in the wins)

Update `tests/style_ratchet.sh` to these values in the same commit as this file; numbers only
go down:

```
error out-params (headers):      0
null-guard writes:               0
std::function in headers:        0
session.h getter pairs:          0
pane-local statics:              0
runtime.cc lines:                ≤ 850
pane header functions:           ≤ 22   → 14 after §2.2
named structs in headers:        ≤ 82   → 40 after §2.3
product LOC:                     ≤ 21300 → ratchet down per punch-list commit; final freeze at 18000 (owner-approved)
camelCase methods in backend/*.h: current count → 0 after §2.1  (add this check)
```

---

## 5. Standing rules (unchanged — these are working; keep checking every diff)

1. **Prime directive:** if the diff wouldn't pass review in `system/camerad/`, don't write it.
   Re-read one exemplar file before each slice.
2. Helpers are born file-local; a header export needs a **second product caller** (tests never
   count). Pane logic worth unit-testing moves to the backend.
3. Errors: `assert()` internal invariants; `std::optional`/empty for parsing; error text only
   where the UI shows a user-actionable message.
4. No registries, observers, callbacks-as-fields, pimpl (beyond §2.4's two), or inheritance.
   Closed sets are static tables. The every-frame redraw IS the notification system.
5. A struct must cross a thread/serialization/subsystem boundary to live in a header.
6. No accessor that is `return member_;` — public members.
7. One dialect: snake_case, short names, no subsystem prefixes on file-local symbols.
8. Ports translate at the boundary: take algorithms and hard-won comments; leave signals,
   singletons, wrappers, naming.
9. Slices pre-budget LOC in PROGRESS; >2× budget is a design smell to fix before commit.
10. Done = diff reads like camerad. Reread it asking "which lines would loggerd not have?"
11. The main file orchestrates; it does not implement.
12. Comment constraints, never narration: one line per lock; why-comments on ordering/lifetime;
    banners in long files; nothing else.
13. Per-frame work is O(visible), never O(dataset).

**Don't overcorrect:** repetition across panes is house style (no widget frameworks beyond
`input_text_with_hint`); deleted abstractions become dumb arrays and asserts, not clever types;
no caches the HUD didn't ask for; the "real complexity" (store staging/drain, scheduler,
split tree, chunked jobs) stays.
