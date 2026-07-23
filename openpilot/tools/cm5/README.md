# CM5 passive dashcam bring-up

This runtime records one V4L2 road camera and raw CAN from one USB Panda in
openpilot's normal segmented route format. It deliberately does not start
registration, upload, UI, model, localization, vehicle identification, planning,
or control processes. It does retain local process diagnostics through
`logmessaged` so failures remain visible in both local logs and recorded rlogs.

## Build

Use a 64-bit Ubuntu 24.04 image, active cooling, and eMMC or NVMe storage. Start
with a USB UVC camera; CSI/libcamera support is a later camera-backend task.

Initialize and build the checkout:

```bash
git submodule update --init --recursive
tools/setup_dependencies.sh
source .venv/bin/activate
scons -u --dashcam
```

The `dashcam` project extra installs the webcam backend's PyAV and headless
OpenCV dependencies as part of dependency setup. Unlike the full openpilot
setup, this path does not download the neural-model Git LFS assets.
The backend requests 1280x720 at 20 FPS from V4L2, limits capture buffering, and
publishes at openpilot's 20 Hz road-camera rate. Confirm that exact mode is
stable on the selected UVC camera before extending the runtime.

The CM5 has no onboard hardware video encoder, so the dashcam build encodes both
video streams in software with H.264. The 1280x720 main stream defaults to
5 Mbit/s with the low-latency `libx264` `ultrafast` preset; the qcamera stream
uses openpilot's 256 kbit/s setting. For compatibility with existing route
tools, the main MPEG-TS file retains openpilot's historical `fcamera.hevc`
filename even though its codec is H.264. Tune CPU/quality on the CM5 with
`--encoder-preset` and `--main-bitrate` after measuring dropped frames and
temperature. At the default bitrates, allow roughly 2.4 GB per recording hour
before log and container overhead.

This follows Raspberry Pi's documented Pi 5/CM5 media path: BCM2712 provides
hardware HEVC *decode*, while recording uses software `libx264`. See Raspberry Pi's
[BCM2712 documentation](https://www.raspberrypi.com/documentation/computers/processors.html#bcm2712)
and [H.264 encoding performance note](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010033-WP-1-H.264%20encoding%20performance%20on%20Raspberry%20Pi%205_series%20computers.pdf).

Confirm the camera and USB Panda are visible:

```bash
v4l2-ctl --list-devices
python - <<'PY'
from panda import Panda
print(Panda.list(usb_only=True))
PY
```

The dashcam build intentionally does not build or flash Panda firmware, so use
a **Red Panda** already flashed with firmware compatible with this checkout.
Other Panda types and bootstub mode are rejected before recording. Power-cycle
the Panda before a passive session; the runtime refuses to start if any of its
per-bus transmit or forwarding counters are already nonzero. For a
manual development session, install either the upstream Panda USB rules or the
restricted service rules in `openpilot/tools/cm5/udev`.

The default arbitration bitrates are 500/500/500 kbit/s and CAN-FD data
bitrates are 2000/2000/2000 kbit/s for Panda buses 0/1/2. These are not
auto-detected. Set `--can-speeds`, `--can-data-speeds`, and, only when required,
`--canfd-non-iso-buses` for the target vehicle. A wrong bitrate can leave the
USB device healthy while silently missing bus traffic. Arbitration rates must
be one of 10/20/50/100/125/250/500/1000 kbit/s; CAN-FD data rates additionally
support 2000 and 5000 kbit/s.

Choose a persistent recording directory. The example below assumes an NVMe
filesystem mounted at `/mnt/dashcam`:

```bash
mkdir -p /mnt/dashcam/realdata /mnt/dashcam/params
python -m openpilot.system.manager.dashcam \
  --camera /dev/v4l/by-id/your-road-camera \
  --log-root /mnt/dashcam/realdata \
  --params-root /mnt/dashcam/params \
  --require-mount /mnt/dashcam \
  --can-speeds 500,500,500 \
  --can-data-speeds 2000,2000,2000 \
  --required-can-buses 0
```

Stop with `Ctrl-C`, `SIGTERM`, or `SIGPWR`. The supervisor stops retention,
forwards `SIGPWR` to `loggerd`, finalizes and fsyncs video and logs before
removing their completion locks, and only then stops camera and CAN publishers.
Recorder children run in separate process groups so a terminal `Ctrl-C` reaches
the supervisor first and cannot bypass that ordering. A singleton lock prevents
two dashcam supervisors from using the same Params root.
The supervisor also watches `roadCameraState`, both `roadEncodeData` and
`qRoadEncodeData`, and `pandaStates`; a live-but-stalled camera, encoder, or
Panda process causes the runtime to exit so systemd can restart it. Loggerd's
completion locks and the offline validator separately prove those encoded
packets were written and indexed.
Any abnormal supervisor exit sends loggerd a dedicated fault signal, so the
active and immediately preceding segments remain locked even if the failure
straddled a rotation or the failing process's final diagnostic message could
not be delivered.
Set `--required-can-buses` to the bus indexes that must be present on the target
vehicle. The runtime then restarts if any required bus is absent during startup
or silent longer than `--can-stale-timeout`. Leaving the list empty is useful on
a bench with an intentionally quiet bus, but cannot detect every wrong-bitrate
configuration.

The retention process deletes the oldest unlocked segments when free space
falls below either 10 GiB or 10%. Change those reserves with `--min-free-gb` and
`--min-free-percent`. Routes may therefore begin at a segment number greater
than zero; the validator reports these as trimmed retained suffixes. On restart,
any segment still carrying writer locks is moved under
`realdata/.interrupted/` instead of being treated as complete.
Loggerd deliberately keeps that lock when required video writers or the
CAN/Panda health stream are missing or discontinuous.

## Unattended systemd installation

The supplied unit assumes this checkout is `/opt/openpilot-cm5`, recordings are
mounted at `/mnt/dashcam`, and the service account is `openpilot-dashcam`.
Build the checkout and its `.venv` in that location before enabling the unit.
Create the account and directories before installing the unit:

```bash
sudo useradd --system --user-group --groups video \
  --home-dir /var/lib/openpilot-cm5 --create-home \
  --shell /usr/sbin/nologin openpilot-dashcam
sudo install -d -o openpilot-dashcam -g openpilot-dashcam -m 0750 \
  /mnt/dashcam/realdata /var/lib/openpilot-cm5/params

sudo install -m 0644 openpilot/tools/cm5/udev/60-openpilot-cm5.rules \
  /etc/udev/rules.d/60-openpilot-cm5.rules
# setup_dependencies.sh installs broad development rules. They are not suitable
# for the unattended service; the CM5 rule uses final group/mode assignments.
sudo rm -f /etc/udev/rules.d/11-openpilot.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

sudo install -m 0644 openpilot/tools/cm5/systemd/openpilot-cm5-dashcam.env \
  /etc/default/openpilot-cm5-dashcam
sudo install -m 0644 openpilot/tools/cm5/systemd/openpilot-cm5-dashcam.service \
  /etc/systemd/system/openpilot-cm5-dashcam.service
sudoedit /etc/default/openpilot-cm5-dashcam
sudo systemctl daemon-reload
sudo systemctl enable --now openpilot-cm5-dashcam.service
```

Set the stable camera path, target-vehicle CAN/CAN-FD bitrates, encoder, and
optional Panda serial in `/etc/default/openpilot-cm5-dashcam` before enabling
the service. The environment file is required. The unit refuses to start unless
`/mnt/dashcam` is a real mount, restarts failed or stalled data paths, sends
`SIGPWR` for the durable shutdown path, and gives the supervisor 60 seconds for
its ordered shutdown. Inspect it with:

```bash
systemctl status openpilot-cm5-dashcam.service
journalctl -u openpilot-cm5-dashcam.service -f
```

## Passive-CAN boundary

The runtime launches `openpilot.tools.cm5.usb_pandad`, a dedicated publisher that
uses Panda's Python USB API, forces `SafetyModel.silent`, and publishes only
`can`, `pandaStates`, and `peripheralState`. It has no `sendcan` subscriber or CAN
transmit call: the receive-only Panda subclass rejects `can_send`,
`can_send_many`, and every non-SILENT safety mode. SILENT and
`controlsAllowed=false` are read back before the first CAN publication and on
every health sample. Hardware type plus per-bus transmit/forward counters are
also checked. USB receive overflow or a per-bus lost-frame counter is likewise
fatal because CAN completeness can no longer be claimed. Any violation stops
the recorder and keeps the route incomplete. Discovery and configuration may
retry before recording starts, but any USB/session discontinuity after a Panda
has been configured is fatal; systemd then starts a fresh route instead of
silently reconnecting across a possible CAN gap.
The supervisor sets `DASHCAM=1` as an explicit runtime marker and does not launch
any control process. Do not run the normal openpilot manager in the same
namespace.

For a safety-critical receive-only installation, also enforce passivity in Panda
firmware or with receive-isolated CAN hardware.

`SIGPWR` and fsync only help while the CM5 remains powered. Vehicle installation
therefore requires an automotive supply or UPS with enough hold-up time for a
managed stop. Wire its ignition/power-loss indication to `systemctl stop
openpilot-cm5-dashcam.service` (the unit maps that stop to `SIGPWR`) and validate
the actual hold-up time. An abrupt power cut has no software-only durability
guarantee.

## Route acceptance check

After stopping a recording, pass any one of its segment directories to the
validator. It discovers the other segments sharing that route prefix:

```bash
python -m openpilot.tools.cm5.validate_route \
  /mnt/dashcam/realdata/00000000--0123456789--0 \
  --decode-video \
  --required-can-buses 0 \
  --max-can-gap 5
```

The check requires a contiguous retained segment range, readable `rlog.zst` and
`qlog.zst`, nonempty `fcamera.hevc` and `qcamera.ts`, valid ending log sentinels,
passive `initData`, CAN frames, SILENT Panda health states with no gap over two
seconds, Red Panda identity, zero transmit/forward/lost-frame counters, road-camera
messages, full-resolution and qcamera H.264 indexes, and overlapping CAN/camera
monotonic timestamps. `--decode-video` fully decodes both files and compares
their decoded frame counts with the logged indexes; qcamera count and timestamp
coverage must also track the main recording. It rejects leftover lock files
unless `--allow-active` is explicitly supplied. `--ffprobe` remains an alias
for `--decode-video`. When `--required-can-buses` is supplied, each listed bus
must span every segment without a gap larger than `--max-can-gap`.

Successful routes remain directly usable with the tooling in this checkout,
including the container-aware Python `FrameReader` and C++ replay path. Run
analysis from a full development build on a workstation (the CM5 `--dashcam`
build intentionally skips desktop tools).

The `fullH264` encode-index extension is specific to this branch. Older stock
openpilot checkouts can probe the video file itself, but their replay scheduler
does not recognize full-resolution H.264; use this checkout for synchronized
camera replay. For example:

```bash
python -m openpilot.tools.plotjuggler.juggle \
  --can \
  /mnt/dashcam/realdata/00000000--0123456789--0/rlog.zst
```

Before vehicle installation, run at least a two-hour recording and verify every
segment, storage growth, CPU temperature, camera stability, CAN frame counts,
clean ignition-off shutdown, and recovery after an intentionally interrupted
write.
