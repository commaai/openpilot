# Comma four keyboard experiment

Run the current device demo:

```sh
python -m openpilot.tools.ui.mici_keyboard_study_demo --condition calibrated
```

Choose **5 Caps above** under Layouts, choose one index finger or two thumbs,
and press Start. Production callers continue to use the original `MiciKeyboard`.
This is an opt-in experiment; it does not install a new device startup script.

## Current keyboard

The floating letter layout has 123 on the bottom left and Space on the bottom
right. The 123 label includes a permanent Caps icon. On symbols, Caps sits above
abc and #+= sits to its right. Symbols follow the iPhone-style two-page ordering;
underscore is on the second page. The original backspace icon is used.

Layer buttons switch on touch-down. Drag to a character and release to enter it
and return to the original page. Returning to the control under the initial
press keeps the opened page. Other page controls switch only on release, so
moving back and forth across abc/123/#+= does not repeatedly switch pages.

A single Caps tap selects Shift; another tap within 300 ms locks it. Slower taps
toggle Shift. Each release restarts a 700 ms return-to-letters delay. Dragging
from 123 to Caps returns immediately. Only period automatically returns to
letters on the URL trial; ordinary punctuation taps stay on symbols elsewhere.

`mici_keyboard_study_demo` applies the frozen `robust_static_v4` offsets from
`keyboard_study_targets_v4.json`, including the existing l+8 adjustment. Visual
key positions stay fixed. Case shares the same letter offsets; symbol keys use
row-level calibration. No dynamic prediction or online learning is enabled.

The plain `mici_keyboard_study` entrypoint retains the earlier l+8 targets and
animated Caps hint for comparison. Six experimental layouts are available;
layouts 3, 5 and 6 share the current letter geometry.

## Study controls and data

The six prompts cover ordinary text, `openpilot.comma.ai`, a username and a
made-up password. `--phrases N` changes the count by cycling through the prompts.
Keep the prompt count fixed when comparing scores. Next is on the far left;
backspace is on the far right and repeats after a 0.5-second hold.

- Hold Next for three seconds to abort and retain the partial recording.
- Hold the start screen for five seconds to open study settings.
- Settings select stock, calibrated, or random assignment for each session.
- Next person creates a separate recording. Technique must be selected again.
- The finish screen shows WPM, previous best, backspaces and elapsed typing time.
  High scores are separated by condition and prompt set.

Logs default to `/data/keyboard-study` on device and `~/tmp/keyboard-study` on PC;
use `--output-dir` to override. Settings survive restarts. Logging begins at Start
and preserves raw evdev events, both Python touch slots, frame events, gestures,
corrections, trial endings, target geometry and configuration. Multiple events
per frame remain distinct. The baseline types with stock single-contact handling
while recording both contacts. Raw capture does not grab the input device.

An optional `MouseState` observer records polling samples and queue-drop counts.
Stopping observation synchronizes with pending callbacks before closing the
writer. The evdev stream reports its clock, axis metadata and capture errors.
Raw recordings and generated reports are not included in the repository.

## Reports and calibration

Generate a report for one recorded condition at a time:

```sh
python -m openpilot.tools.ui.keyboard_study_report DATA_DIR \
  --output-dir REPORT_DIR --condition CONDITION
```

Add `--keyboard-image SCREENSHOT.png` for interactive and high-resolution tap
heatmaps. Use a 536:240 screenshot matching that condition's actual key positions.
Never overlay stock taps on the rearranged layout or pool different positions.
Reports include incomplete sessions and expose quality flags and capture errors.
The burst detector conservatively flags rapid out-of-prompt sequences; raw files
remain unchanged. A wrong tap's intended key is inferred only from a matching
prompt prefix, not from the key it selected.

The original median calibration can be reproduced with:

```sh
python -m openpilot.tools.ui.keyboard_study_calibrate DATA_DIR --output MODEL.json
```

The constrained optimizer is separate from deployment:

```sh
python -m openpilot.tools.ui.keyboard_study_optimize DATA_DIR OUTPUT_DIR \
  --reports-root ~/tmp --audit-directory ARCHIVED_SAME_LAYOUT_DIR
```

It matches exact letter geometry, uses recorded last-down points, normalizes
recorded offsets to the l+8 comparison baseline, and verifies recorded selections
before replay. Complete sessions train; compatible partial sessions and archived
same-layout recordings are extra checks. Keep deleted/spam recordings excluded
from DATA_DIR. Duplicate sessions in multiple reports are counted once.

The optimizer freezes Space and controls, bounds letter-center changes to 12 px,
balances sessions, and downweights uncertain mistakes. Taps more than one local
key pitch from their inferred target have zero training weight but remain in
error totals. This does not identify a user's true intent. Shift directions must
persist across inner resampling; outer held-out sessions are never used to fit
their own candidate. The model and per-fold results are exported for inspection.

The command never changes runtime targets or deploys a model. A passing gate
permits a trial, not a claim of optimality. Repeated participants may span folds,
and fitting decisions have been informed by earlier diagnostics. Evaluate frozen
candidates on fresh recordings before drawing conclusions from fitted replay.

## Tests

```sh
SCALE=1 python -m unittest discover \
  -s openpilot/system/ui/widgets/tests -p 'test_keyboard_*.py'
```
