# MEB hold fuzzing (DO NOT MERGE)

On-car fuzzing of the VW MEB standstill hold / TSK faults on a gen2 ID.4. The openpilot branch `meb-hold-fuzz`
points opendbc at the `meb-hold-fuzz` branch: #3861 (Daniel's #3846 reverted, plus the minimal hold fix), plus
two variant switches.

## How it works

- `openpilot/selfdrive/car/meb_fuzz.py` runs inside card, right before `CarInterface.apply()`.
  - It replaces only what openpilot sends: `actuators.longControlState`, `actuators.accel` and `latActive` (off
    by default; you steer).
  - The production `MebLongStateMachine` still produces every ACC_18 frame.
  - Opendbc variants it sets each cycle:
    - `hold_source`: `hms` = #3861, or `standstill` = old master (goes inactive on ESC_50.Standstill)
    - `starting`: `master` = drive off on Standstill, or `stock` = drive off once the hold manager holds (#3846)
- It is controlled by `/data/meb_fuzz/control.json`, re-read within 0.5 s of any change. See `CONTROL_EXAMPLE` in
  `meb_fuzz.py`.
- On-screen text goes through `alertDebug`. selfdrived shows it engaged or not (events.py hack).
- Outputs:
  - `/data/meb_fuzz/status.json`: 2 Hz.
  - `/data/meb_fuzz/results.jsonl`: one line per run, with signal change events.
  - `cloudlog` `meb_fuzz` events in the rlog.
  - `state.json`: run counter and restart flag.
- **Run flow:**
  1. Engage (SET) anywhere in the lot.
  2. P control to `setup_speed`.
  3. The program steps run, each until its `until` expression or `timeout`.
  4. Stop and hold until you disengage. Reverse, re-engage for the next run.
- **Aborts:** brake, gas, disengage, TSK 6/7. Too fast, or rolling back over 0.5 m/s, makes it brake to a hold.
- **TSK 7** shows "RESTART CAR" until the TSK is healthy for 2 s after an ignition cycle.

## Live use from a laptop

```sh
# device (ssh comma@<ip>)
cd /data/openpilot && ./openpilot/cereal/messaging/bridge

# laptop, in an openpilot checkout of this branch
ZMQ=1 python openpilot/tools/meb_fuzz/watch.py <ip> --log session.jsonl
python openpilot/tools/meb_fuzz/ctl.py <ip> program openpilot/tools/meb_fuzz/programs/e0_clean_stop.json
python openpilot/tools/meb_fuzz/ctl.py <ip> program openpilot/tools/meb_fuzz/programs/e1_aborted_stop.json --hold-source standstill
python openpilot/tools/meb_fuzz/ctl.py <ip> live stopping -0.4      # drive directly, "live off" to return
python openpilot/tools/meb_fuzz/ctl.py <ip> alert "Pull forward" "then engage"
python openpilot/tools/meb_fuzz/ctl.py <ip> results 3
python openpilot/tools/meb_fuzz/ctl.py <ip> off
```

`watch.py` decodes raw CAN with opendbc:
- hold manager `VMM_02.HMS_Status`: 0 not_holding, 9 stopping, 1 holding, 5 starting, 6 releasing
- `ESC_50.Standstill` / `Motion_State` (1 fwd, 2 reverse, 3 stopped) / `EPB_Status`
- `Motor_51.TSK_Status` (3 regulating, 5 brake_only, 6 temp fault, 7 perm fault)
- camera `ACC_18.ACC_Status_ACC`
- our `ACC_18` request (0 none, 1 HALTEN, 4 ANFAHREN, 5 RAMP), accel and `ACC_Anhalten`

## Programs

Each step is `{"state", "accel", "until", "timeout"}`. `state` / `accel` / `until` may be expressions over
`v a hms ss motion tsk cam req inhibit gear t T gas brake` (`t` = s in step, `T` = s in run).

- `e0_clean_stop`: baseline stop and hold.
- `e1_aborted_stop` / `e1b`: hold engages → go (RAMP releases it) → stopped without a hold → go → stop. This is
  the sequence before the TSK=7 faults. Run with `--hold-source standstill` to reproduce the old inactive-without-
  hold step.
- `e2_churn_at_stop` / `e5_creep_churn`: pid/stopping chatter like the real policy oscillation.
- `e3_halten_from_1p5`: hold request from 1.5 m/s.
- `e4_release_when_held`: go right after the hold engages; A/B `--starting master` (RAMP) vs `--starting stock`
  (ANFAHREN).
