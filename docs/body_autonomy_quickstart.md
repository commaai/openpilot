# Comma Body Autonomy v0 (Attention-Gated Pursuit) Quickstart

This implementation adds a runnable autonomy daemon (`bodyautonomyd`) and teleop web controls for an indoor, low-speed prototype.

## What is implemented

- New daemon: `tools.bodyteleop.autonomyd`
  - Publishes `testJoystick` commands.
  - Runs FSM states (`idle`, `acquire`, `frozen`, `advance`, `lost`).
  - Uses `driverStateV2` eye/face probabilities as an attention gate.
  - Uses runtime params for target visibility, bearing, and obstacle distance.
  - Plays sound cues on acquire/loss transitions.
- Web API/UI controls in `tools.bodyteleop.web` + static UI:
  - Enable/disable autonomy.
  - Set target visibility / target distance / target bearing / obstacle distance.
  - Observe live autonomy status.

## Runtime parameters

- `BodyAutonomyEnabled` (bool)
- `BodyAutonomyTargetVisible` (bool)
- `BodyAutonomyTargetDistance` (float meters)
- `BodyAutonomyTargetBearingDeg` (float degrees)
- `BodyAutonomyObstacleDistance` (float meters)
- `BodyAutonomyStatus` (json status payload)

## Basic flow

1. Start manager in not-car mode with web joystick as usual.
2. Open body teleop UI.
3. Click **Enable** in the new autonomy panel.
4. Tune target/obstacle values to validate behavior.
5. Observe `state` and output joystick axes in status line.

## Notes

- This is a v0 control stack scaffold. Depth/perception ingestion should replace parameter-fed target/obstacle values in the next iteration.
- Keep speed low and always use supervised testing with an emergency stop.
