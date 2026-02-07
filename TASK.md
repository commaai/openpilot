goal: determine optimal qcamera encoder settings for great dashcam video with tiny file sizes

---

qcams are the low resolution camera files logged in system/loggerd/.
file size is very important since we upload them over an expensive cellular connection.
we want the best quality dashcam footagae for the tiniest file sizes.

ffmpeg makes beautiful qcams for small file sizes, but we probably need to stick to the hardware encoder on the Snapdragon 845 to make it run realtime.

- the kernel driver for the encoder is available in .context/agnos-kernel-sdm845/
- you might find some extra settings in the kernel driver that may be useful
  - we should test different bitrates, both h264 and hevc, and anything else you think is relevant
- we are ok spending more encoder time, compute, power, etc to achieve our goals! we are also ok spending some more CPU time if that would help!
- SSH into the comma four at comma@comma-d3c3ad4b to run things

---

your output should be a tool that sweeps over the encoder settings with various input video files that are re-encoded with the new settings.
the tool outputs a beautiful single page HTML report with all videos for me to compare. ideally you can also supplement the report with some objective metrics,
so that i can choose the right settings.
i will supply the input videos.
the tool will need to build a special version of encoderd that takes the input video and re-encodes with your specified settings, then you'll need to SSH into a
comma four that I will give you to build the tool, run it, and get the output.

---

Refinement: qcamera encoder sweep scope and acceptance criteria
 Objective
Find one final qcamera encoder preset that delivers the best visual dashcam quality per byte, with current production qcamera settings treated as the quality floor and included as the baseline in all comparisons.
 Constraints
- Realtime is required on comma four (`>= 1.0x` throughput at target FPS, no sustained frame drops).
- FPS must remain fixed to current qcamera FPS.
- Resolution is tunable.
- Evaluate both H.264 and HEVC equally.
- Packaging/container format is flexible for sweep outputs (`.ts`, `.mp4`, or elementary streams are all acceptable).
- Tuning may include userspace, kernel/V4L2, and pipeline changes (including `encoderd`/`loggerd`-related code if needed).
- CPU/power/thermal metrics are not required.
 Deliverable
Build a sweep tool that:
- Builds and runs a special `encoderd` variant on comma four.
- Re-encodes provided input clips across a preset matrix.
- Supports checkpoint/resume for interrupted runs.
- Writes all artifacts to filesystem.
- Produces a single-page HTML report with synchronized playback, baseline clearly highlighted, and objective metrics.
 Report requirements
Include at least:
- Codec, resolution, GOP/B-frame structure, bitrate/rate-control settings.
- Output file size and effective bitrate.
- Objective quality metrics (VMAF, SSIM, PSNR) versus source clip.
- Realtime pass/fail and measured encode speed ratio.
- Ranked candidates plus a Pareto view (quality vs size).
 Input corpus requirements
Use short clips to keep runtime to a few minutes:
- 8-10 clips, each 8-12 seconds.
- Scenarios: bright highway, bright city, dusk/sun glare, night urban, night highway low light, rain/wet windshield, high-motion/vibration, and high-detail signage/texture.
- Use high-quality source clips (not previously qcamera-compressed clips).
 Final output decision
Provide:
- A recommended single best preset.
- A short Pareto shortlist of near-optimal alternatives.
- Clear rationale for why the recommended preset is the best tradeoff.
