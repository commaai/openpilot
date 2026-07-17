# fallback2 handoff — 2026-07-16

## UPDATE 2026-07-17: bench deployment on 192.168.61.224 (in progress)

Commits reworked to a minimal diff after review: `ac40394ae2` "modeld: run big model in
separate process" (+104/-42, everything in modeld.py, one line in cereal/services.py,
no new files, big_worker() entered via `modeld.py --big-worker`) and `dfacba99cf`
"promote to big only after proven on-time outputs" (promotion needs 20 consecutive
frames of fresh worker outputs, prevents latching on the worker's cold first frames).

Deployed to `comma-e3b715f2` at 192.168.61.224 (user-directed; it has the eGPU,
add1:0001 at 5 Gbps on bus 4-1). The old fallback bench `.96` re-leased to
192.168.62.67 (usbgpu-test harness rig, no eGPU attached). Deployment is the two
files ported onto the device's own lineage in /data/openpilot (a "reboot recovery"
lineage, dd55ef573e) — python-only overlay, prebuilt stays valid, backup of replaced
bits in /data/continue.sh.goal6.bak and /data/goal6-local-mods-backup.patch.

Gotchas hit, all resolved:

- This lineage's params_keys.h lacks `UsbGpuActive` — modeld crash-loops on put_bool
  (UnknownKeyName) and every crash spawns another worker. Added the key and rebuilt
  just `scons openpilot/common/params_pyx.so` (small, safe).
- The workstation `bench_ctl.py` jungle replay was wedged on a stale USB handle
  (running but zero CAN delivered, all `can` events empty, card blocked forever in
  "waiting for CAN"). Kill + relaunch fixed it. It must be launched via
  `python -c 'import panda, opendbc; exec(...)'` because ~/openpilot/panda shadows the
  venv panda package. Never `pkill -f` patterns that appear in your own ssh cmdline.
- Leftover goal6 experiment processes (tmux supervisor) stopped comma.service mid-test
  and had rebooted the device earlier; tmux killed, `hotplug-watch.service` (a passive
  eGPU state logger from /data/goal6-perfect) stopped.
- watch_fallback.py measured gaps by its own receive time; its 1 Hz /proc scan showed
  up as phantom 90 ms modelV2 gaps every second. Rewritten to measure publish-time
  (`sm.logMonoTime`) deltas — phantom gaps vanished.
- After a worker is SIGKILLed mid-USB-operation the next AMD init can fail with
  "BL not ready" (PSP bootloader timeout) even though the USB device enumerates fine.
  Device reboot recovers it (this bench's goal6 lineage exists for exactly that).

**Measured result so far (the core claim): while the big model loads in the isolated
worker, small publishes clean 20 Hz — max publish gap 51 ms, p99 51 ms,
frameDropPerc 0.0.** Zero lagging during load by the alert's own metric.

Also validated live: worker death latches to small exactly once ("big model failed,
latching to small"), modeld keeps publishing, no crash.

Still pending on the bench: full boot-to-promotion sequence with the streak fix,
software-unplug latch test (`echo 4-1 > /sys/bus/usb/drivers/usb/unbind`), no
re-promotion check, soak.

### Session end state (2026-07-17, user stand-down)

- The device kernel (`4.9.103 #4 SMP PREEMPT Wed Jul 1`, custom batman@docker build)
  intermittently hard-lockups CPU 0 during the big model's USB load:
  `Kernel panic - not syncing: Watchdog detected hard LOCKUP on cpu 0` in
  /sys/fs/pstore/dmesg-ramoops-0, crash-looping the device roughly 2 min into each
  load. One load DID complete cleanly earlier (05:37, "big model ready"), so it is
  flaky, not deterministic. The other usbgpu-test rig validates kernel tuple
  `#1 SMP PREEMPT Sun Jul 12` — this looks like a known-bad vs known-good kernel
  matrix that parallel sessions (tmux: agnos = kernel, codex = USB transfer bundles,
  codex2 = ASM2464 firmware) are working through. The fallback layer itself was never
  the crash cause.
- Something on the workstation ssh'd in at 06:23 and ran `systemctl stop comma`
  (source 192.168.43.51 = this workstation, all other agent sessions idle at the
  time) — unattributed, watch for it when resuming.
- Bench quiesced on user request: device comma+fbwatch stopped (device then took
  another watchdog reboot, expected to come back idle), workstation bench_ctl killed,
  jungle ignition commanded off, all background monitors stopped.
- To resume: memory notes `bench-61-224` and `fallback2-process-isolation` have the
  full recipes; branch `fallback2` carries the code and `scripts/fallback-bench/`
  carries the tooling and this knowledge.

## UPDATE 2026-07-16: process-isolated big model implemented (commit `7b0fdc8cc8`)

New candidate solution committed on local `fallback2` (not pushed): the big model now
loads AND runs in a separate `usbmodeld` process instead of a thread.

Root-cause reasoning it is built on:

- The load lag was a GIL problem, not a core problem. The loader thread holds the GIL
  for long stretches (huge pkl unpickle, USB weight upload), stalling the RT inference
  thread regardless of CPU affinity.
- The core-6 NaN is consistent with process-global tinygrad state being raced once true
  cross-core parallelism was allowed. Same-core GIL serialization masked it on core 7.
- A separate process removes both couplings at once: separate GIL, separate tinygrad
  state, and the kernel guarantees SCHED_OTHER usbmodeld can never preempt SCHED_FIFO
  modeld on core 7. Both models stay on core 7.

Architecture (`selfdrive/modeld/usbmodeld.py` + `BigModelHandle` in `modeld.py`):

- modeld keeps the small model and stays the only publisher. Every frame it broadcasts
  its inputs (frame_id, both transforms, desire/traffic_convention/action_t) on
  `customReservedRawData0` (logged, small).
- usbmodeld (spawned by modeld, PDEATHSIG-tied, SCHED_OTHER on core 7) opens its own
  VisionIPC clients, loads + warms the big model, then answers each request with
  pickled parsed outputs on `customReservedRawData1` (not logged, 2MB queue).
- Promotion: modeld drains child outputs non-blocking; once outputs track current
  frame_ids (within 2 frames) and the car is disengaged, it switches to waiting up to
  45 ms per frame for the child's output.
- Fallback: on timeout, modeld runs small on the same frame (one late publish, no gap).
  Latch to small permanently on child death, any non-finite output, or 3 consecutive
  misses. Latching kills the child. `UsbGpuActive` reflects state as before.
- modeld never blocks on the child before promotion, so zero added lag during load by
  construction. During load the child gets only modeld's leftover core-7 cycles
  (RT throttling guarantees >=5%), so loading is slower but harmless.

NOT bench-validated: both `.96` and `.204` were unreachable (no ping / no route) this
session. Required bench validation, in order, with the event-level monitor running:

1. Baseline master small-only max inter-frame gap trace.
2. Boot with eGPU: verify no `Driving Model Lagging` and max gap comparable to baseline
   for the whole load/warmup, then promotion only while disengaged, `UsbGpuActive`
   True, 20 Hz big output, modelV2 frame ids continuous across the switch.
3. Unplug eGPU while big active: exactly one failure event, same-frame small output,
   `UsbGpuActive` False, no repeated promotion, no gap > 75 ms, at most ~1-2 dropped
   vipc frames (frameDropPerc must stay under 1%).
4. Replug: no re-promotion (latched until modeld restart).
5. Long soak: hours on big, watch for spurious misses (GC pauses in the child are the
   main suspect; if seen, tune BIG_RESPONSE_TIMEOUT/BIG_MAX_MISSES or disable GC in the
   child after warmup).

`bench-tools/watch_fallback.py` remains valid (still keyed on `UsbGpuActive`).

## Executive state

The USB-eGPU big/small model fallback is **not car-ready**.

What is currently established:

- Small-model inference works normally before big becomes active.
- Big-model inference works normally after it has loaded and warmed.
- A catchable big-model USB failure can switch back to small without killing `modeld`.
- The failure latch fix prevents selecting the failed big model again on every frame.
- Loading/warming big in the background on core 7 causes a visible `Driving Model Lagging` interval.
- Moving only the loader/warmup thread to core 6 did not solve the system safely; the test produced `speed: nan` immediately. That candidate is rejected.
- There is no validated configuration that provides continuous, valid small-model output during background big-model construction/warmup in the same `modeld` process.

Do not use the current fallback implementation for driving until loading and handover are validated with event-level gap monitoring and valid model outputs.

## Repositories and branches

### Main fallback worktree

- Path: `/home/batman/openpilot-fallback2`
- Checked-out branch: `fallback2`
- Local HEAD: `5766463910` — `Revert "modeld: isolate big model loading"`
- Remote tracking tip: `origin/fallback2` at `ee79ba17d2` — the rejected core-6 experiment
- Local branch is one commit ahead of `origin/fallback2`; the revert is not pushed.

Relevant history, newest first:

```text
5766463910 Revert "modeld: isolate big model loading"
ee79ba17d2 modeld: isolate big model loading
d25dd7d117 modeld: latch big model failure
25864df435 modeld: resume small model on big failure
4ef4ffc03b Revert "modeld: keep small inference alive during big model"
55d48689c3 modeld: keep small inference alive during big model
09a24fecb6 Revert "modeld: avoid fallback load starvation"
399968e889 modeld: avoid fallback load starvation
dded98946b ui: show model status
f8280e8aa9 modeld: add USB GPU fallback
```

Important: `origin/fallback2` currently contains the rejected core-6 commit. The local revert `5766463910` must be pushed or otherwise applied before anyone treats the remote branch as the current state.

### Clean experimental branch

- Branch: `fallback2-minimal`
- Tip: `716fb3ffad` — core-6 loader experiment, rejected
- Its clean sequence was:

```text
716fb3ffad modeld: isolate big model loading
5a9bfc1dce modeld: latch big model failure
c029ef11a0 modeld: resume small model on big failure
dded98946b ui: show model status
f8280e8aa9 modeld: add USB GPU fallback
```

This branch is useful for reviewing the minimal individual commits, but its tip must not be tested in a car.

### `openpilot7`

- Path: `/home/batman/openpilot7`
- Branch: `master`
- HEAD: `60716edc3752339a1c83e38745dd5835c8b93060`
- It was not updated with the fallback fixes.
- It has a modified submodule entry: `opendbc_repo` (`git status` shows `m opendbc_repo`). Preserve that existing state.

## Exact fallback implementation state

The relevant code is in `openpilot/selfdrive/modeld/modeld.py`.

The original fallback implementation starts small, then creates and warms big in a daemon thread. Promotion to big is allowed only while `carControl.enabled` is false.

`25864df435` / `c029ef11a0` changed exception fallback so that:

- A big inference exception selects `small_model` immediately.
- `UsbGpuActive` becomes false immediately.
- Small inference runs on the same received camera frame instead of dropping that frame.
- The old behavior that withheld publication until disengagement was removed.

`d25dd7d117` / `5a9bfc1dce` added the required failure latch to promotion:

```python
elif not big_failed and not sm["carControl"].enabled and big_model is not None and model is not big_model:
```

Without `not big_failed`, unplugging the eGPU caused this loop every 50 ms:

1. Big inference raised a USB exception.
2. The exception handler ran small and published successfully.
3. The next loop promoted big again because the bench was disengaged.
4. Big failed again.

This repeated exception behavior was directly observed on the bench. The latch commit fixes that specific bug.

The rejected core-6 commit changed only:

```python
set_core_affinity([7])
```

to:

```python
set_core_affinity([6])
```

inside `load_big_model()`. The local revert restores core 7.

## Bench evidence

Device restriction used throughout: **only `192.168.62.96`, over SSH; never ADB**.

The Jungle/bench controller was already running and sending CAN. Later work intentionally did not start, stop, or control it.

The device-side Corolla bench launch environment was configured in `/data/continue.sh` as:

```bash
export AGNOS_VERSION=18.4
export XDG_CACHE_HOME=/data/tg_cache
export FINGERPRINT=TOYOTA_COROLLA_TSS2
export SKIP_FW_QUERY=1
export TESTING_CLOSET=1
```

Observed after setup:

- `CarParams=TOYOTA_COROLLA_TSS2`
- CAN alive and frequency-valid
- `pandaStates` alive and frequency-valid
- `carState` alive and frequency-valid
- Ignition was on from the existing bench feed

### Nominal small and big results

After repairing the build artifacts:

- Small during/after startup published approximately 19.25–20.0 Hz in coarse 20-second samples.
- Once big was active, `UsbGpuActive=True`.
- Big published exactly 20.0 Hz over a 20-second sample.
- `modelV2` was alive and frequency-valid.
- The bench UI nevertheless showed `Driving Model Lagging` during big load. The coarse average-rate measurement missed the transient maximum inter-frame gap and was therefore inadequate.

The UI observation is authoritative evidence that the load path still violates the requirement even though the long-window average remained near 20 Hz.

### Physical unplug result on commit `c029ef11a0`

With big active, the eGPU was physically unplugged on the bench.

Observed:

- tinygrad raised a catchable USB assertion:

```text
AssertionError: read(0xA8F0, 8) failed: -4
```

- `modeld` remained alive.
- `UsbGpuActive` became false.
- Small published at 20.0 Hz in the subsequent 20-second sample.
- Because promotion lacked `not big_failed`, the big USB exception repeated every 50 ms even though small continued publishing. This led to the latch fix.

The latch fix was deployed as `5a9bfc1dce`, but a full replug/load/unplug validation of the latched behavior was interrupted before completion.

### Core-6 result

The hypothesis was that moving only background big load/warmup to core 6 would stop it starving active small inference on core 7.

The reported test result was immediate `speed: nan`. This is a hard invalid-output failure; the candidate is rejected.

There were no device logs collected for this failure because `.96` became unreachable (`No route to host`) before inspection. Therefore the exact root cause is not proven.

The strongest current hypothesis is unsafe concurrency in shared tinygrad/process state: CPU affinity separates scheduling but does not isolate process-global JIT caches, allocators, device objects, queues, or QCOM/USB runtime resources. Moving the loader to another CPU may permit more concurrent mutation rather than making it safe. Treat this as a hypothesis pending logs, not a confirmed root cause.

## Build/deployment problems encountered

The device was switched among historical commits and accumulated incompatible/stale native and generated artifacts.

Observed issues included:

- Native extension ABI errors involving `params_pyx.so` and `visionipc_pyx.so`.
- A full parallel SCons build exhausting memory/rebooting while compiling the big model.
- A 6 GiB `/data/swapfile` was created, but swap did not persist across reboot.
- The `prebuilt` marker had been moved to `/data/prebuilt-minimal-serial` during a serial build attempt and was not restored after reboot.
- Without `/data/openpilot/prebuilt`, each service restart re-entered SCons.
- `openpilot/selfdrive/modeld/models/tg_input_devices.json` was left as a zero-byte file, causing both `modeld` and `dmonitoringmodeld` to crash with `JSONDecodeError`.

The generated device map was repaired with:

```bash
cd /data/openpilot
PATH=/usr/local/venv/bin:$PATH PYTHONPATH=/data/openpilot \
  scons -j1 openpilot/selfdrive/modeld/models/tg_input_devices.json
```

The valid file was 224 bytes and described QCOM defaults plus AMD for the USB-GPU queue. The prebuilt marker was restored to `/data/openpilot/prebuilt`, after which the tree showed clean and services launched normally.

Do not interpret those build failures as evidence about the fallback source logic; they were deployment/build-state failures. A future deployment should start from a clean checkout or verified artifact set and avoid parallel cold big-model compilation on-device.

## Monitoring

The old `/home/batman/bench-tools/watch_fallback.py` was misleading because it sampled at roughly 1 Hz and inferred readiness from obsolete `bigmodeld` log strings/parameters. It also referenced `UsbGpuFailed`, which is not a valid Params key in this branch.

It has been rewritten locally to event-level monitoring and passes ruff and Python syntax checks. It now:

- Polls messaging every 20 ms.
- Records every unique `modelV2` frame arrival.
- Emits a `lag` event for any inter-message gap greater than 75 ms.
- Records phase transitions among `small_loading`, `big`, and `small_fallback` using `UsbGpuActive`.
- Emits one-second summaries with maximum and p99 gap, message age, execution time, enabled state, and modeld process/core information.
- Writes JSONL to `/data/fallback_watch.jsonl` when run on-device.

The improved monitor was validated locally but was not successfully copied/launched on `.96` because the device went offline. The last attempted SCP failed before changing the device.

Suggested launch sequence once the device is reachable:

```bash
scp -i ~/.ssh/comma_ed25519 /home/batman/bench-tools/watch_fallback.py \
  comma@192.168.62.96:/tmp/watch_fallback.py

ssh -i ~/.ssh/comma_ed25519 comma@192.168.62.96 '
  pkill -f /tmp/watch_fallback.py 2>/dev/null || true
  : > /data/fallback_watch.jsonl
  cd /data/openpilot
  setsid env PYTHONPATH=/data/openpilot /usr/local/venv/bin/python \
    /tmp/watch_fallback.py >/tmp/watch_fallback.log 2>&1 < /dev/null &
'
```

Start the monitor before restarting only the device service so it captures the complete small startup, big load/warmup, and promotion. Do not manipulate the Jungle/controller.

## Last known device state

- Device: `192.168.62.96`
- Access policy: SSH only
- Last confirmed deployed source before loss of connectivity: `5a9bfc1dce` on the clean minimal history
- Last confirmed tinygrad revision: stock `e6fbede1576f0a201e0b2c112499552a61280abd`
- Last confirmed model state before the final experiment: big active and publishing at 20 Hz
- The core-6 candidate deployment/test history after that is not independently confirmed from device logs.
- At the end of the session the device had no ping and SSH returned `No route to host`/timeouts.

## What is actually working

The narrowest useful working code state is:

- Original `f8280e8aa9` USB-GPU fallback and UI commit.
- Immediate same-frame small execution after a catchable big inference failure (`25864df435`/`c029ef11a0`).
- Failure-latched promotion (`d25dd7d117`/`5a9bfc1dce`).
- Loader affinity restored to core 7 by local revert `5766463910`.

This state fixes the repeated-failure selection bug, but **does not fix model lag during big loading**. It is not a complete product solution.

## Remaining design constraint

The observed results are:

```text
background load on core 7 -> valid output, visible transient model lag
background load on core 6 -> reported immediate NaN output
big already loaded         -> valid 20 Hz output
catchable big USB failure  -> small can publish, latch required
```

A reliable solution likely needs one of:

1. Real runtime/process isolation for big construction and warmup, with a safe way to transfer or retain the ready model.
2. Explicit tinygrad synchronization/isolation changes proven safe for concurrent model construction and inference.
3. No concurrent loading: run small only, or accept a controlled no-output interval while replacing it with big. The latter violates the stated no-lag requirement.

CPU affinity alone is not sufficient evidence of isolation.

## Safety and next steps

1. Do not car-test `ee79ba17d2` or `716fb3ffad`; they contain the rejected core-6 experiment.
2. Push/apply local revert `5766463910` if `origin/fallback2` must be made non-NaN again.
3. Recover `.96` connectivity and collect its exact HEAD and logs before any further conclusion about the NaN failure.
4. Deploy and start the event-level monitor before any service restart.
5. Establish a baseline master/small-only maximum-gap trace.
6. Reproduce core-7 loading with the event monitor and capture the precise worst gap and timing.
7. Do not propose another car test until the full load and handover trace has no unacceptable gaps, no NaNs, and valid `modelV2` throughout.
8. Test fallback only on the bench: activate big, physically remove eGPU, verify exactly one failure event, `UsbGpuActive=False`, continuous valid small output, no repeated promotion, and stable maximum inter-frame gap.

