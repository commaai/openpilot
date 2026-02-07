# QCamera Encoder Sweep Plan

## Objective
Find one final qcamera preset with the best visual quality per byte, while meeting realtime on comma four and treating current production qcamera settings as a hard baseline in every comparison.

## User-provided run constraints (locked)
- Source clips currently provided:
  - `/home/batman/sixpilot/5beb9b58bd12b691_000000cd--1ffb321983--5--fcamera.hevc`
  - `/home/batman/sixpilot/5beb9b58bd12b691_000000ce--af935294c3--13--fcamera.hevc`
- Treat these as one day clip and one night clip for the current sweep.
- Each source is ~60s; use only a `5-10s` excerpt per clip (default plan: `8s`).
- SSH to comma four is confirmed working.
- Artifact destination defaults in this plan are approved.
- Hard output limit: each encoded output file must be `<= 10 MB`.
- Audio can be skipped for now.

## Non-negotiable acceptance criteria
- Realtime: `encode_speed_ratio >= 1.0` at fixed qcamera FPS (`20`) with no sustained frame drops.
- Baseline included: production qcamera settings are always present and clearly highlighted.
- Compare H.264 and HEVC on equal footing.
- Report: single-page HTML with synchronized playback, metrics, ranking, and Pareto plot.
- Decision output: one recommended preset, short Pareto shortlist, and written rationale.
- Size cap: every `(preset, clip)` encoded artifact is `<= 10 MB`.

## Known baseline from current code
- Current qcamera preset (from `system/loggerd/loggerd.h`):
  - codec: `QCAMERA_H264`
  - resolution: `526x330`
  - bitrate: `256000`
  - gop: `15`
  - b-frames: `0`
  - fps: `20`
- Current encoder controls set in `system/loggerd/encoder/v4l_encoder.cc` include bitrate, GOP/P/B frame counts, rate control (`VBR_CFR`), profile, and IDR period.

## Deliverable structure
Create a new tool directory:

```
tools/qcam_sweep/
  README.md
  presets/
    coarse.yaml
    refine.yaml
  run_sweep.py
  feed_clip.py
  compute_metrics.py
  build_report.py
  select_recommendation.py
  remote_run.py
  templates/
    report_template.html
```

Sweep outputs:

```
artifacts/qcam_sweep/<run_id>/
  manifest.json
  jobs.jsonl
  clips/<clip_id>/
  encoded/<preset_id>/<clip_id>.<ext>
  metrics/<preset_id>/<clip_id>.json
  report.html
  recommendation.md
```

## Work packages (subagent-executable)

### SA-1: Baseline + control inventory
- Owner type: `explore`
- Goal: lock baseline and enumerate tunable V4L2/MSM VIDC controls with legal ranges.
- Inputs:
  - `system/loggerd/loggerd.h`
  - `system/loggerd/encoder/v4l_encoder.cc`
  - `.context/agnos-kernel-sdm845/include/uapi/linux/v4l2-controls.h`
- Actions:
  - Extract baseline values and current hardcoded controls.
  - Build a short allowlist of candidate sweep knobs (`bitrate`, `resolution`, `gop`, `b_frames`, `rc_mode`, optional qp bounds if supported).
  - Mark each knob as `safe`, `experimental`, or `skip`.
- Output artifact: `tools/qcam_sweep/docs/control_inventory.md`.
- Done when: control inventory lists exact IDs/enums and initial sweep ranges.

### SA-2: Special encoderd sweep hooks
- Owner type: `general`
- Depends on: SA-1
- Goal: make qcamera encoding tunable at runtime without changing default behavior.
- Actions:
  - Add a sweep config path for qcamera settings (JSON or env-backed), read only when sweep mode is enabled.
  - Preserve default behavior when no sweep config is provided.
  - Allow codec toggle (`h264` or `hevc`) and tunable bitrate/GOP/B-frames/rate-control/resolution.
  - Ensure output filename extension can vary by codec/container for sweep artifacts.
- Code touchpoints:
  - `system/loggerd/encoderd.cc`
  - `system/loggerd/loggerd.h`
  - `system/loggerd/encoder/v4l_encoder.cc`
  - `system/loggerd/SConscript` (only if a dedicated `encoderd_sweep` binary is added)
- Done when:
  - `encoderd` (or `encoderd_sweep`) accepts a per-run qcamera config.
  - Existing loggerd/encoderd tests still pass with default settings.

### SA-3: Clip feeder and run harness
- Owner type: `general`
- Can start in parallel with SA-2 after interfaces are agreed.
- Goal: feed input clips into VisionIPC as `camerad` stream and orchestrate encode runs.
- Actions:
  - Add clip-trim preprocessing to create `8s` (configurable) excerpts from provided 60s inputs.
  - Implement `feed_clip.py` to decode source clips to `nv12` and publish frames at fixed FPS over `VisionIpcServer("camerad")`.
  - Implement `run_sweep.py` to run `loggerd + encoderd + feed_clip` for each `(preset, clip)` job.
  - Capture output file, frame count, filesize, effective bitrate, elapsed time, and speed ratio.
  - Mark job failed if output file exceeds `10 MB`.
  - Add checkpoint/resume via `jobs.jsonl` state machine (`pending/running/done/failed`).
- Done when:
  - Interrupted runs restart and skip completed jobs.
  - One command can execute a matrix over all clips and leave artifacts on disk.

### SA-4: Metrics + report generation
- Owner type: `general`
- Depends on: SA-3 outputs
- Goal: compute objective quality and produce an interactive single-page report.
- Actions:
  - Implement `compute_metrics.py` for VMAF/SSIM/PSNR against source clip.
  - Normalize comparison by matching frame rate and resolution before metric computation.
  - Implement `build_report.py` to generate one HTML page with:
    - synchronized playback controls,
    - baseline highlighted in all tables/charts,
    - per-clip and aggregate metrics,
    - realtime pass/fail,
    - size-cap pass/fail (`<=10 MB`),
    - ranked list and Pareto scatter (`quality` vs `size`).
- Done when:
  - `report.html` is standalone and viewable locally.
  - Every encoded artifact has a metrics JSON row (or explicit failure reason).

### SA-5: Remote execution wrapper for comma four
- Owner type: `general`
- Depends on: SA-2, SA-3
- Goal: run sweep end-to-end through SSH and retrieve outputs.
- Actions:
  - Implement `remote_run.py` to:
    - build on device,
    - launch sweep on device,
    - support resume,
    - pull artifacts back.
  - Target host defaults to `comma@comma-d3c3ad4b`, overridable by CLI.
- Done when:
  - One command executes remote sweep and returns local artifact directory.

### SA-6: Selection and final recommendation
- Owner type: `general`
- Depends on: SA-4
- Goal: produce final preset recommendation with transparent logic.
- Actions:
  - Implement `select_recommendation.py`:
    - Filter to realtime-pass candidates.
    - Enforce baseline quality floor (aggregate VMAF/SSIM/PSNR not worse than baseline by tolerance).
    - Rank by smallest size, tie-break by quality.
    - Compute Pareto shortlist.
  - Write `recommendation.md` with winner + near-optimal alternatives.
- Done when:
  - Recommendation is reproducible from metrics data alone.

## Preset matrix (concrete starting point)

### Phase A: coarse sweep (fast prune)
- Clips: 2 provided clips (day + night), each trimmed to `8s`.
- Presets: baseline + 9 candidates.

Suggested IDs:
- `baseline_h264_526x330_256k_g15`
- `h264_526x330_192k_g15`
- `h264_526x330_224k_g15`
- `h264_526x330_256k_g30`
- `h264_640x400_320k_g20`
- `hevc_526x330_160k_g15`
- `hevc_526x330_192k_g15`
- `hevc_526x330_224k_g20`
- `hevc_640x400_256k_g20`
- `hevc_640x400_320k_g30`

### Phase B: refine sweep (decision quality)
- Clips: full 8-10 clip corpus from TASK.md scenarios (deferred until more source clips are provided).
- Presets: baseline + top 5-7 from Phase A + up to 2 experimental qp/rate-control variants if supported by SA-1 inventory.

## Runtime and pass/fail rules
- Expected frames per clip: `round(duration_sec * 20)`.
- Realtime pass:
  - `encode_speed_ratio >= 1.0`, and
  - frame drop rate <= `0.5%`.
- Size pass:
  - output file size `<= 10 * 1024 * 1024` bytes.
- Quality floor relative to baseline (aggregate):
  - `VMAF >= baseline_vmaf - 1.0`
  - `SSIM >= baseline_ssim - 0.005`
  - `PSNR >= baseline_psnr - 0.3 dB`

## Input corpus contract
- Current run: 2 clips (day/night), each trimmed to `8s` from provided 60s sources.
- Final recommendation target: expand to `8-10` clips, each `8-12` seconds, covering all TASK.md scenarios.
- Manifest file (`clips_manifest.yaml`) fields:
  - `clip_id`
  - `path`
  - `scenario` (must cover all required scenarios)
  - `notes` (optional)

Current `clips_manifest.yaml` seed entries:
- `clip_id: day_01`, `path: /home/batman/sixpilot/5beb9b58bd12b691_000000cd--1ffb321983--5--fcamera.hevc`, `scenario: day`
- `clip_id: night_01`, `path: /home/batman/sixpilot/5beb9b58bd12b691_000000ce--af935294c3--13--fcamera.hevc`, `scenario: night`

## Recommended command flow
Local smoke test:

```bash
python tools/qcam_sweep/run_sweep.py \
  --clips clips_manifest.yaml \
  --presets tools/qcam_sweep/presets/coarse.yaml \
  --out artifacts/qcam_sweep/smoke \
  --max-jobs 4 \
  --resume
python tools/qcam_sweep/compute_metrics.py --run artifacts/qcam_sweep/smoke
python tools/qcam_sweep/build_report.py --run artifacts/qcam_sweep/smoke
python tools/qcam_sweep/select_recommendation.py --run artifacts/qcam_sweep/smoke
```

Trim step (current two 60s inputs -> 8s excerpts):

```bash
ffmpeg -y -ss 00:00:20 -i /home/batman/sixpilot/5beb9b58bd12b691_000000cd--1ffb321983--5--fcamera.hevc -t 8 -c copy artifacts/qcam_sweep/day_01_8s.hevc
ffmpeg -y -ss 00:00:20 -i /home/batman/sixpilot/5beb9b58bd12b691_000000ce--af935294c3--13--fcamera.hevc -t 8 -c copy artifacts/qcam_sweep/night_01_8s.hevc
```

Remote full run:

```bash
python tools/qcam_sweep/remote_run.py \
  --device comma@comma-d3c3ad4b \
  --clips clips_manifest.yaml \
  --presets tools/qcam_sweep/presets/refine.yaml \
  --out artifacts/qcam_sweep/final \
  --resume
python tools/qcam_sweep/compute_metrics.py --run artifacts/qcam_sweep/final
python tools/qcam_sweep/build_report.py --run artifacts/qcam_sweep/final
python tools/qcam_sweep/select_recommendation.py --run artifacts/qcam_sweep/final
```

## Parallelization plan
- Parallel now:
  - SA-1 and SA-3 scaffolding can start immediately.
  - SA-4 can scaffold report format using mock data while SA-3 is in progress.
- Sequential gates:
  - SA-2 must land before full sweep.
  - SA-6 runs only after metrics for all non-failed jobs are complete.

## Definition of done
- Tool can sweep presets over user-provided clips with checkpoint/resume.
- Full artifact set is on disk and reproducible from manifest + presets.
- `report.html` includes synchronized videos, baseline highlight, objective metrics, realtime status, ranking, and Pareto view.
- `recommendation.md` states one winner and a short Pareto shortlist with rationale.
