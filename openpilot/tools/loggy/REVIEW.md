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

### 2.1 The naming rename never happened (v1 B5)

Backend headers are still cabana camelCase (`canEvents`, `canEventSummary`, `assignSources`,
`addSignal`, `beginFrame`…) while shell/panes are snake_case. This is the one v1 item that was
skipped rather than done. It's ~30 method names, purely mechanical, and it's the last thing
making the backend read as "ported" rather than "native". Do it in one sweep commit.

### 2.2 Layering inversion around `panes/messages.h`

`panes/messages.h` is the one pane header that still exports an API: `MessageSummary`,
`parse_message_id_state`, `initial_message_id_for_store`, and all four CSV builders — and it is
included by `panes/binary.cc`, `panes/signal.cc`, `panes/historylog.cc`, `panes/computed.cc`,
and **`shell/workspace.cc`**. The shell including a pane header is upside down, and
`backend/export.h` survives as a 3-line shim pointing at it.

Fix: these are backend utilities — move `MessageSummary` + message-id state helpers into the
store or a small `backend/csv.{h,cc}` alongside the CSV builders; delete the `export.h` shim;
messages.h becomes a one-liner like the other twelve. (`kLoggySeriesPathPayload` in browser.h
and `map_basemap_effective_cache_root` in map.h are single small exceptions — fine.)

### 2.3 Header structs grew: 50 → 82 (v1 A7 relocated instead of shrinking)

Moving pane helpers to the backend moved their structs into backend headers. Some are now
legitimate boundary types; `backend/route.h` (13 structs) and `backend/video.h` (8) deserve an
audit against the rule: *a struct earns a header name by crossing a thread, serialization, or
subsystem boundary*. Everything with one caller goes file-local. Target: ≤ 40.

### 2.4 Pimpl came back (v1 A3): 3 instances

`backend/video.h` (×2) and `backend/panda_live.h`. Defensible motive (keeps FFmpeg/libusb out
of every include), but note `loggerd/video_writer.h` includes FFmpeg directly — the house
answer. Either inline the members, or keep pimpl **only** for the ffmpeg/libusb pair and say so
in a one-line comment. Don't let it spread past these two.

### 2.5 LOC: 21,241 product lines — **revised target: ≤ 18,000** (owner-approved 2026-07-05)

The original ≤15k target predates the full feature surface (workspace engine, live-source
matrix, computed-series editor). Adeeb has signed off on **18k** as the final product-LOC
budget; it supersedes the plan's §2.14 number. Current split: panes 9,173 / backend 6,140 /
dbc 1,658 / shell 4,270 / main 239 (tests excluded).

Route there (~3.2k to cut, no capability loss): §2.2 layering fix, §2.3 single-caller struct
inlining, per-pane boilerplate dedup in the worst lines-per-feature panes, and tightening
`backend/live.cc` (840) and `backend/route.cc` (659). Do NOT squeeze past 18k at the cost of
readability or features — that violates "don't overcorrect". Ratchet accordingly: product LOC
baseline steps down with each punch-list commit and freezes at 18,000.

### 2.6 HUD p99 is a lifetime max, not a window

Settled camera CPU is 5.69 ms but the HUD still shows p99 28.9 ms from the startup decode
storm forever. Make p99 a rolling window (~5 s) so the gate number is readable at a glance —
and so regressions aren't hidden inside a stale startup spike (nor good steady-state hidden
behind one).

---

## 3. Functional items NOT fixed (retested this pass)

1. **Message-row selection still needs two clicks** to drive Binary/History/Signal (retested:
   single click on a row leaves Binary pinned to the old ID). Qt cabana follows on single
   click. This was in the v1 appendix and did not get a slice. It's a one-line-ish selection
   condition, and it breaks 8 years of cabana muscle memory.
2. **`tools/cabana/panda.cc` is still modified** (reference tree, +2/−1). The fix itself is
   fine; it needs Adeeb's explicit sign-off or a revert + loggy-side workaround, per the plan's
   "old tools stay untouched" rule. Flag it in the next summary to the owner.
3. **Default presets are missing the panes that define each tool's feel** (owner feedback
   2026-07-05, after driving both launchers). The camera panes work — they're just absent from
   the preset JSONs:
   - `layouts/cabana.json`, Cabana tab: currently messages/historylog/signal/binary only. Real
     cabana leads with **video top-right and charts bottom-right** — add a camera pane and a
     plot pane to the tab (video over plot on the right column matches the original).
   - `layouts/jotpluggler.json`, Jotpluggler tab: currently browser/plot/logs/map — add a
     camera pane (real jotpluggler always shows the road camera preview).
   Verify by launching both launchers with `--demo` and comparing against the reference
   binaries side by side; this plus item 1 closes most of the remaining "feels like" gap.

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
