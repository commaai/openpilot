# Indoor "SCP-173-style" Comma Body v2 + comma 4 Plan (Safety-First)

## 0) Goal, constraints, and what "good" looks like

This document describes a practical architecture to run on the `good-remote-body` branch of openpilot for an indoor mobile robot that:

- tracks one or more humans,
- only advances toward a target when your "attention gate" says it is allowed,
- avoids furniture/obstacles using depth sensing,
- and plays a sound effect when it acquires a target.

> **Safety boundary:** treat this as an **indoor novelty/interaction robot** running at low speed in controlled spaces with participant consent, geofencing, emergency stop, and conservative collision margins.

Success criteria for v1:

1. Detect and track a person at 0.8-4.0 m in typical indoor lighting.
2. Determine "attention state" (looking/not-looking, eyes open/closed proxy).
3. Advance only when `attention_state == NOT_ATTENDING`.
4. Re-plan around static/dynamic obstacles (chairs, table legs, people) while preserving a no-contact safety envelope.
5. Trigger audio cue on target acquisition and mode transitions.

---

## 1) Recommended stack (CTO-level decision)

Use a **two-layer architecture**:

- **Layer A (on comma 4 / openpilot process graph):** perception, behavior state machine, and command publication.
- **Layer B (on body base controller):** low-level velocity control and hard safety interlocks.

### Why this stack

- Reuses existing openpilot messaging/process patterns instead of introducing a large external middleware migration.
- Keeps hard-stop and motor safety close to actuators.
- Allows you to iterate behavior logic quickly in Python while keeping deterministic low-level control in the base firmware/controller.

### Core software components

1. **Perception node (Python + ONNX/TFLite runtime):**
   - person detection/tracking,
   - face/head pose + blink/eye-state estimation,
   - depth-based occupancy map generation.
2. **Behavior node (state machine):**
   - explicit SCP-like finite states (`IDLE`, `ACQUIRE`, `FROZEN`, `ADVANCE`, `LOST`).
3. **Navigation node:**
   - local planner over depth occupancy (DWA/TEB-style local planning or lattice + costmap).
4. **Audio node:**
   - plays one-shot effects on event bus triggers.
5. **Safety supervisor:**
   - speed caps, stop-distance checks, watchdog, E-stop.

---

## 2) Algorithms by requirement

## A. Human detection + tracking

Use a detector + tracker combo:

- **Detector:** YOLOv8n / MobileNet-SSD (person class only) at ~10-20 Hz.
- **Tracker:** ByteTrack or DeepSORT for stable IDs.

Outputs per frame:

- `target_id`, `bbox`, `confidence`, `range_estimate`, `bearing`.

### B. "Not paying attention" gating

Implement an attention score from three signals:

1. **Head yaw/pitch** from face landmarks or lightweight head-pose model.
2. **Gaze approximation** (if eye landmarks are available).
3. **Blink/eye closure proxy** via EAR (Eye Aspect Ratio) over a short temporal window.

Define a robust gate (example):

- `ATTENDING` if head/gaze within angular threshold for >300 ms.
- `NOT_ATTENDING` if gaze off-target for >500 ms **or** eyes-closed proxy for >400 ms.
- Use hysteresis to avoid oscillation.

### C. Depth sensing + obstacle avoidance

Depth input options:

- native depth camera on the robot base, or
- stereo/depth-from-motion pipeline if available from sensor suite.

Pipeline:

1. Depth frame -> point cloud / depth image filtering.
2. Ground segmentation + obstacle extraction.
3. 2D local occupancy grid (e.g., 4 m x 4 m around robot).
4. Inflation radius for robot footprint + safety buffer.
5. Local planner computes collision-free velocity command to target bearing.

Planner recommendation for v1 indoor:

- **DWA-like local planner** with penalties for proximity, high angular jerk, and dynamic obstacles.
- Fallback emergency stop if nearest obstacle distance < hard threshold.

### D. Behavior control (SCP-like movement)

Finite state machine:

- `IDLE`: no target / waiting.
- `ACQUIRE`: detect + lock target, play "found target" sound once.
- `FROZEN`: attention indicates target is watching -> command zero velocity.
- `ADVANCE`: target not attending -> planner allowed to move.
- `LOST`: target not visible; rotate/search slowly within safety constraints.

This makes behavior explainable and debuggable.

### E. Audio cueing

Event-driven sound playback:

- `TARGET_ACQUIRED` -> play SFX once.
- Optional: `STATE_CHANGE` cues (subtle short effects).

Use non-blocking playback process/thread to avoid perception latency.

---

## 3) Integration in openpilot (good-remote-body branch)

## Proposed services/modules

1. `selfdrive/body/perceptiond.py`
   - camera ingest + detector/tracker + attention estimation + depth occupancy.
2. `selfdrive/body/behaviord.py`
   - state machine + policy gate + target management.
3. `selfdrive/body/navd.py`
   - local planning and motion command generation.
4. `selfdrive/body/audiod.py`
   - sound effects triggered by events.
5. `selfdrive/body/safetyd.py`
   - command validation, clamps, watchdog integration.

## Message schema additions (conceptual)

Add/extend pub-sub messages for:

- `humanTrackState`: tracks + confidences + selected target.
- `attentionState`: attending/not-attending + confidence + timing.
- `localOccupancy`: compact occupancy/depth hazard summary.
- `bodyBehaviorState`: current FSM state + reason.
- `bodyMotionCommand`: desired `v, omega`.
- `bodySafetyStatus`: stop reason / watchdog / estop.

## Process manager wiring

- Register new daemons in manager config (conditional on body hardware mode).
- Ensure fail-closed startup ordering:
  - no motion until `perceptiond`, `navd`, and `safetyd` healthy.

---

## 4) Hardware recommendations (v1 practical)

Minimum reliable indoor platform:

- **Compute:** comma 4 (primary behavior/perception host).
- **Base:** comma body v2 motor base/controller.
- **Depth sensor:** short-range RGB-D module with good indoor IR behavior.
- **Audio:** small powered speaker.
- **Safety I/O:** physical E-stop button that cuts motor enable line.
- **Optional:** bumper switches or ToF ring for near-field failsafe.

---

## 5) Development roadmap (8-10 weeks)

## Phase 1 (Week 1-2): bring-up + telemetry

- Validate camera/depth streams and timing.
- Build recording/replay for indoor scenes.
- Add dashboards/logging for FPS, latency, dropped frames.

Exit criteria:

- Stable sensor ingest at target FPS and synchronized timestamps.

## Phase 2 (Week 2-4): person tracking + attention estimation

- Implement detector+tracker and target lock logic.
- Add head pose/eye-state proxies with temporal smoothing.
- Tune attention hysteresis on recorded scenarios.

Exit criteria:

- >90% stable target lock in test scenes.
- Attention false-switch rate below target threshold.

## Phase 3 (Week 4-6): depth occupancy + local planner

- Build local costmap from depth.
- Implement DWA-like planner and obstacle inflation.
- Add hard-stop envelope and watchdog.

Exit criteria:

- No contacts in obstacle-course tests (chairs, tables, moving people).

## Phase 4 (Week 6-8): behavior FSM + audio + system integration

- Implement full FSM and transitions.
- Event-based sound playback.
- End-to-end integration under process manager.

Exit criteria:

- Reliable SCP-style freeze/advance behavior in controlled room tests.

## Phase 5 (Week 8-10): robustness + acceptance

- Stress test lighting, clutter, multiple people, occlusions.
- Tune speed limits and failover behavior.
- Produce operator checklist and safety SOP.

---

## 6) Safety envelope (non-optional)

1. **Top speed cap:** start very low indoors (e.g., walking-speed fraction).
2. **Dynamic stop distance:** velocity-dependent minimum clearance.
3. **Dual-stop strategy:** software E-stop + hardware power cut.
4. **Human override priority:** any manual command preempts autonomy.
5. **Geofence/no-go zones:** never enter stairs, fragile zones, door thresholds.
6. **Session arming flow:** explicit arm/disarm, audible status cue, timeout auto-disarm.
7. **Data/privacy:** local processing preferred; consent for any recording.

---

## 7) Suggested model/runtime choices

For real-time edge performance:

- Start with lightweight models (quantized int8 where possible).
- Use ONNX Runtime / TFLite with profiling on target hardware.
- Keep end-to-end perception loop budget to ~50-80 ms initially.

Fallback strategy:

- If face/eyes are not confidently visible, default to conservative `ATTENDING` (freeze).

---

## 8) Test plan (what to measure)

Key metrics:

- Target acquisition latency.
- Attention gate precision/recall.
- Min obstacle clearance distribution.
- Planner oscillation rate.
- Safety-stop trigger count and causes.
- End-to-end command latency (sensor -> motor command).

Test scenarios:

1. Single person, clean room.
2. Cluttered room with chairs/table legs.
3. Crossing person occlusion.
4. Low light / backlight.
5. Multiple humans (target selection stress).
6. Sudden obstacle insertion near path.

---

## 9) Concrete implementation checklist in this repo

1. Create `selfdrive/body/` daemons listed above.
2. Add manager process entries and startup guards.
3. Define/extend messages for tracks, attention, occupancy, behavior, commands.
4. Implement replay tooling for offline tuning.
5. Add integration tests for FSM transitions and safety clamps.
6. Add runbook in docs for calibration, arming, and emergency procedures.

---

## 10) Open questions to finalize before implementation

1. Which exact depth sensor is available on your body v2 setup?
2. What max speed/acceleration limits are acceptable for your environment?
3. Single-target only or multi-target prioritization rules?
4. Is eye-closure required strictly, or is "not facing robot" enough for v1?
5. Do you need operation fully offline (no network dependencies)?
6. What speaker hardware/audio output path is available on your build?

Once these are answered, we can produce a sprint-by-sprint execution board with file-level tasks and acceptance tests.
