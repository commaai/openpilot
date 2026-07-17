# eGPU fallback bench control

Runbook for the big/small fallback bench: a comma device (mici hardware) with an eGPU
(ASM2464PD bridge + AMD GPU, USB `add1:0001`, normally bus `4-1` at 5 Gbps), fed CAN by
a panda jungle attached to the workstation, replaying a Corolla TSS2 route.

Addresses are deliberately not hardcoded — use whatever device/jungle is attached:

```bash
export BENCH_DEV=<device ip>   # find it: scan the bench subnets for ssh answering
                               # the comma key, eGPU rig has add1:0001 in /sys/bus/usb
```

ssh as `comma@$BENCH_DEV` with key `~/.ssh/comma_ed25519`. The jungle is auto-detected
(first attached jungle).

## 0. Before touching anything: check who owns the bench

Several agent sessions share this device (kernel work in tmux `agnos`, USB link work in
`codex`, bridge firmware in `codex2`). Signs someone else is driving it:

```bash
ssh comma@${BENCH_DEV:?set BENCH_DEV to the bench device IP} 'sudo journalctl -b --no-pager | grep -E "sudo.*COMMAND" | tail -10'
```

Look for `systemctl stop comma`, `qmp_dump.py`, `tee /sys/class/usbpd/usbpd0/hard_reset`
from a PWD that is not yours, minutes old. If present, coordinate before proceeding —
parallel sessions WILL stop comma.service under your test. Workstation side:
`tmux list-sessions` and peek with `tmux capture-pane -pt <name>`.

## 1. CAN replay (workstation)

Exactly ONE bench_ctl instance, launched with the panda-import workaround
(`~/openpilot/panda` shadows the venv package):

```bash
: > /tmp/bench_ctl.log
setsid nohup /home/batman/openpilot/.venv/bin/python -c \
  'import panda, opendbc; exec(compile(open("/home/batman/bench-tools/bench_ctl.py","rb").read(), "/home/batman/bench-tools/bench_ctl.py", "exec"), {"__name__":"__main__"})' \
  > /tmp/bench_ctl.log 2>&1 < /dev/null &
```

Controls: `echo 1|0 > /tmp/bench_ign` (ignition), `echo engage|disengage > /tmp/bench_cruise`,
`engage.sh` retries rising edges until the device reports enabled.

Wedge symptom: device receives `can` events but ALL EMPTY (`len(msg.can)==0`), card blocks
forever in "Waiting for CAN". Fix: kill bench_ctl, relaunch. It wedges on a stale USB
handle when devices reboot. WARNING: never `pkill -f <pattern>` where the pattern appears
in your own ssh/bash command line — it kills your own shell (use the loop over
`pgrep -f bench_ctl\.py` from a command line that doesn't contain that string, or kill by
saved pid).

## 2. Device stack

`/data/continue.sh` must be:

```bash
#!/usr/bin/env bash
export AGNOS_VERSION=18.4
sudo sh -c "echo -1 > /proc/sys/kernel/sched_rt_runtime_us" || true
cd /data/openpilot
exec ./launch_openpilot.sh
```

- RT throttling must stay disabled (`-1`) — the default 95% cap stalls the RT stack ~50 ms
  every second.
- Bench env lives in the repo's `launch_env.sh` (SKIP_FW_QUERY=1,
  FINGERPRINT=TOYOTA_COROLLA_TSS2, TESTING_CLOSET=1, XDG_CACHE_HOME=/data/tg_cache).
- The deployed tree needs `UsbGpuActive` in `openpilot/common/params_keys.h`; if the
  lineage lacks it: add the line and `scons openpilot/common/params_pyx.so` (small, safe).
- Keep the `prebuilt` marker or every restart re-enters scons.

Start/stop: `sudo systemctl restart comma` / `sudo systemctl stop comma`
(`reset-failed` first if it shows failed).

## 3. Event watcher

```bash
scp watch_fallback.py comma@${BENCH_DEV:?set BENCH_DEV to the bench device IP}:/data/watch_fallback.py
ssh comma@${BENCH_DEV:?set BENCH_DEV to the bench device IP} '
  sudo systemctl reset-failed fbwatch 2>/dev/null; sudo systemctl stop fbwatch 2>/dev/null
  : > /data/fallback_watch.jsonl
  sudo systemd-run -q --unit=fbwatch --collect -p User=comma -p WorkingDirectory=/data/openpilot \
    -p Restart=always -E PYTHONPATH=/data/openpilot bash -c "/usr/local/venv/bin/python /data/watch_fallback.py"'
```

Writes JSONL to `/data/fallback_watch.jsonl`: `lag` events for publish-time gaps > 75 ms
(measured from `modelV2.logMonoTime`, NOT receive time — receive-time measurement gives
phantom 1 Hz gaps from the watcher's own /proc scan), 1 Hz `summary` with max/p99 gap and
`frame_drop_perc` (the `Driving Model Lagging` alert fires at frameDropPerc > 1), `phase`
transitions from `UsbGpuActive`. The transient systemd unit does not survive reboot —
restart it after every boot, before restarting comma.

## 4. eGPU health and recovery ladder

Check: `cat /sys/bus/usb/devices/4-1/idVendor` == `add1`, `speed` == `5000`,
`bConfigurationValue` == `1`. NEVER run `lsusb -v` against it — descriptor dumps can
crash the xHCI. In escalating order when the big model worker fails:

1. "BL not ready" (PSP bootloader timeout) after a previous GPU user was SIGKILLed:
   reboot the device (`sudo reboot`).
2. "TLP error after retries" (PCIe transactions dead through the bridge):
   `cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python \
   /data/goal6-perfect/amd_reset_rd2.py full` — bridge-level PCIe power cycle + RXPLL
   reset + secondary bus reset, then a test big-model load.
3. Bridge won't answer vendor control transfers (`read(0xCC37) failed`): USB-PD hard
   reset `echo 1 | sudo tee /sys/class/usbpd/usbpd0/hard_reset`, wait ~15 s for
   re-enumeration; last resort is a physical DC pull (~5 s) on the eGPU.
4. Known substrate issue: kernel `4.9.103 #4 (Jul 1)` intermittently hard-lockups CPU 0
   during the big model's sustained USB load (`Watchdog detected hard LOCKUP on cpu 0` in
   /sys/fs/pstore/dmesg-ramoops-0) and the device reboots itself. Not a fallback bug —
   retry the load; the validated usbgpu-test kernel tuple is `#1 (Jul 12)`.

## 5. Fallback validation procedure

1. Watcher up (step 3), CAN replaying with ignition on (step 1), then
   `sudo systemctl restart comma`.
2. Load phase: modeld logs `models loaded in Xs` (small), spawns
   `modeld.py --big-worker`. PASS: `frame_drop_perc` stays < 1 and no sustained publish
   gaps while the worker loads (takes ~4 min — the worker only gets modeld's leftover
   core-7 cycles, that is the design).
3. Promotion: worker logs `big model ready`; modeld promotes only after 20 consecutive
   fresh outputs while disengaged, logs `big model active`, `UsbGpuActive` -> true,
   watcher phase -> `big`.
4. Engaged gate: use `engage.sh`; while enabled, promotion must not happen; after
   disengage it must.
5. Failure: with big active, software-unplug
   `echo 4-1 | sudo tee /sys/bus/usb/drivers/usb/unbind`. PASS: exactly one
   `big model failed, latching to small`, same-frame small publish (no gap > ~75 ms
   beyond one late frame), `UsbGpuActive` -> false, phase -> `small_fallback`, no
   re-promotion afterwards (rebind with `.../bind` and confirm it stays latched).
6. Soak: hours on big, watch for spurious misses (worker GC pauses are the suspect —
   tune the 45 ms deadline / 3-miss latch in modeld.py if seen).

## 6. Handing the bench back

```bash
ssh comma@${BENCH_DEV:?set BENCH_DEV to the bench device IP} 'sudo systemctl stop fbwatch comma'
echo 0 > /tmp/bench_ign          # before killing bench_ctl
kill <bench_ctl pid>
```

Restore points for the pre-fallback2 device state: `/data/continue.sh.goal6.bak` and
`/data/goal6-local-mods-backup.patch` on the device.
