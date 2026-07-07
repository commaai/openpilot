# Style review — v5 (2026-07-06, owner manual review)

**Owner manual-review pass.** Adeeb reviewed both presets line-by-line; this batch matches the
references properly instead of the earlier over-engineered chrome. Commits `9ef62643c` (load
frame-drops + first parity round), `5aa575e83` (Opus behavior-diff round: input map, palette,
white canvas, split-to-empty-plot, close X), `7d724152a` (manual-review pass). Delivered 15 of
16 review items, each GUI-verified under Xvfb:

- Camera: cover-crop default, no engaged/seg-frame-t overlays, no "View" label.
- Browser: jotpluggler density (Field+Value, no sparkline/path), layout selector combo,
  Map/Camera special-source draggables, edge drag-to-split, ImGui Demo removed.
- Plots: peak-preserving decimation + RangeFit (Y no longer jitters), cursor-follow scroll,
  toolbar/chips moved into the right-click menu, jotpluggler palette/legend/white canvas.
- Panes: centered corner close X; a split makes an empty plot.
- Cabana: per-signal plot/delete buttons; **plot materializes a decoded /dbc/ signal series**
  (the unified-namespace capability — decoded signals are now plottable and browsable);
  binary-view message-history chips; message editor collapses so the signal list is primary.
- Shell: Ctrl+Z undoes DBC-edit-then-workspace; parallel shutdown for a snappy exit.

**Enum/state-block plots — now landed (was deferred #6).** Enum names are propagated through the
extract pipeline (`SegmentExtractResult.metadata`, previously dropped in route.cc) via a new
ingest drain (`RouteIngestor::drain_enum_metadata`, same staged-under-mutex / drained-once pattern
as timeline spans and logs) into a session enum registry keyed by series path. DBC value
descriptions (`Signal::val_desc`) feed the same registry in `materialize_decoded_signal`, so
decoded-CAN signals with value tables also render as blocks. A pane whose every series is an enum
switches to jotpluggler's state-block renderer (`build_state_blocks` / `state_block_color` /
`state_block_label` / `draw_state_blocks` ported into plot.cc): one labelled colored lane per
series, Y pinned to [0,1] with decorations off, hover tooltip showing value/name/range/duration.
A scale/offset/derivative/python transform disqualifies a series (the value→name map no longer
holds), so those fall back to lines. GUI-verified on the demo route: `/carState/gearShifter`→"drive"
and `/selfdriveState/state`→colored disabled/enabled lanes, while `/carState/vEgo` stays a line.

**Deferred (documented):** still open from round-3, the exports timestamp/atomicity cluster.

Historical v4/v3/v2 content follows for the audit trail.

# Style review — FINAL (v4, 2026-07-06 late)

**Ship verdict (v4).** The red-team recovery push landed as six root-reviewed commits on top of
the v3 state (`4cafc47fc`…`6a18736b3`): data honesty (route duration snaps to real content end,
925.10s not 960.00s; deprecated series extracted; CSV export isolated from sibling-pane
selection), visual identity (owner-directed `struct Theme` in shell/theme.h with
kLightTheme/kDarculaTheme, Light default; binary per-signal colored spans + M/L markers;
camera/plot chrome cleanup), playhead semantics (Binary/Signal/History/Browser/Plot-legend all
bound to the tracker via O(log n) summaries, never the chart zoom; playback autostarts on
load), interaction polish (real splitter drags with undo, binary edge-resize semantics matching
Qt cabana, plot empty-series persistence, quiet map failure banner), a camera-decode fix
(frames keep publishing when playback outpaces decode; blank-canvas starvation eliminated), and
theme-token completion (map carto palette themed — it used to stay dark in light mode; zero
color literals outside theme.cc, color_rgb() deleted). Every batch was line-reviewed at the
root, hand-merged where waves collided (binary.cc three-way: signal spans × recency heat), and
GUI-verified under Xvfb in both themes. Ratchet current: LOC 21,669 · structs 85 · runtime.cc
912 · color literals 0 — all with dated justifications in tests/style_ratchet.sh.

**Red-team round 2 (2026-07-06, pre-owner-review).** 8-finder adversarial fan-out + root GUI
pass: 29 raw findings, 23 survived per-finding refutation, all triaged at the root. Fixed in
four commits (`6a2abf2db`…`b2454df9c`): History visible at the default preset, decode-lifecycle
races (single-point abort consumption, UI-thread stall in set_camera_index, stale-fill floor,
active-key preservation), timeline None barriers persisted across incremental re-merges, chart
click-seek moved to release (zoom-drag no longer yanks the playhead), one zoom-history entry
per gesture, browser uncapped via a skeleton cache (11k series browsable), camera recovery
after workspace undo, Ctrl+Z/Ctrl+Shift+Z, table + plot-selection theme tokens, luminance-aware
badge text, resize preserves min/max, History rebuild quantized to 4 Hz during playback,
sparse-series sample-hold, autostart yields to user pause, cold basemap cache is a miss, and
the mangled `start_` rename scrubbed from user-visible strings. Accepted deviations (documented,
not fixed): binary heat is per-BYTE recency (Qt cabana is per-bit; per-bit needs a bit-level
store query — owner call whether it's worth it), splitter clamp drift on over-drag (standard
imgui behavior), and History's full-event-copy page rebuild is rate-limited rather than
replaced with a bounded backend tail query (follow-up if it shows up on real rlogs).

Remaining backlog, deliberately deferred, is §3b: timeline hover thumbnails + route-info bar,
layout-save pretty-printing, plan decision #3 (DBC-decoded signals as plottable series paths —
red-team #32), and a cosmetic rename of backend/csv.{h,cc} (it outgrew its name; it houses the
CAN summary/grid queries too). One item still needs the owner: `tools/cabana/panda.cc` carries
a +2/−1 libusb error-handling fix in the reference tree.

Historical v3/v2 content follows for the audit trail.

# Style review — v3 (2026-07-06)

**Ship verdict.** The completion push (baseline `1976165e5` → ship, ~25 commits) closed every
phase of the plan: style punch list, LOC settled at the measured floor, 7-slice parity audit
with 21/24 defects fixed, perf gates green on a quiet box (steady playback 5.3 ms CPU, heavy
6-plot workspace p99 8.2 ms, TTFU < 2 s), and an independent fresh-eyes review over the full
diff concluded it "reads like system/ code" (its one should-fix — a regression test that
didn't discriminate the two DBCManager generation counters — is fixed). Ratchet frozen at the
shipped values; it runs in every smoke. Remaining backlog, deliberately deferred, is §3b:
timeline hover thumbnails + route-info bar (the one capability gap vs Qt cabana), layout-save
pretty-printing, and the unreproduced transient route-load modal. One item needs the owner:
`tools/cabana/panda.cc` carries a +2/−1 libusb error-handling fix in the reference tree.

Historical v2 content follows for the audit trail.

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

## 3b. Parity audit outcome (Phase 4, 2026-07-06)

Seven-slice GUI audit vs both reference tools found 24 defects (5 blockers). Ledger:
- **21 fixed and verified** across five fix batches + integrator work, notably: playhead-bound
  Messages table with byte-change heatmap, sorting, and suppress toggles (the core cabana feel
  was simply missing); camera seek-storm stale-frame guard + honest overlay time; Find Bits
  statistics corrected (pairing walked the wrong timeline — totals could exceed the candidate's
  own event count); DBC Save As ImGui ID collision; big-endian drag-resize (three compounding
  bugs incl. a pre-existing hover-exclusivity failure affecting ALL leftward drags); dirty-edit
  protection with per-signal pending state; undo scoped to a new `file_set_generation()`
  (distinct from the mutation `generation()` caches key on); preset rebalances; series removal;
  arrow-key stepping; Light-theme tokens for hand-drawn fills.
- **Deferred (feature backlog):** timeline hover thumbnails + route-info bar (the one true
  capability gap vs Qt cabana). Layout save pretty-printing (saves currently write a one-line
  float-noise dump). Signal pane's fixed-height internal table should flex with the pane.
- **Unreproducible, watching:** one transient "Failed to load route" modal over a loaded route
  (no spontaneous-modal code path exists; suspected genuine transient segment failure).
- **Perf note:** integrator killed three per-frame O(dataset) patterns the fixes introduced or
  exposed (heatmap event copies → in-place store query; per-frame row rebuild → generation-keyed
  skeletons; history page rebuild → input-keyed cache). Authoritative frame numbers pending the
  quiet-box Phase-5 audit — all measurements during the fix wave were on a load-30+ box.
- **Process:** `workspace_smoke` fails on fresh worktrees when the scons cache restores
  `generated_dbcs/.stamp` without its side-effect files — fix the SConscript side-effect
  declaration at ship. Agent GUI sessions must use `--settings` isolation (shared user settings
  got polluted with test DBCs and a theme flip).

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
