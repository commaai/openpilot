# GPS relay between two commas

The car has a metallized (athermic) windshield. It attenuates GPS badly enough that
the device behind it tracks **zero satellites** — not a weak fix, nothing at all. The
fix is a second comma somewhere with sky view, relaying its position to the main one
over the network.

Downstream nothing changes: `parknavd`, the UI and `hardwared` all keep subscribing
to `gpsLocationExternal` and cannot tell the fix arrived over wifi.

---

## Vocabulary

- **GPS device** — the spare comma with sky view. Supplies the fix.
- **Main device** — the one in the car. Consumes the fix.

Throughout, substitute your own addresses for `GPS_IP` and `MAIN_IP`.

---

## Setup

### 0. Prerequisites

Both devices need:

- branch `gps-relay` checked out at `/data/openpilot`
- SSH reachable as `comma@<ip>`
- to be on the same network, able to reach each other (not just your laptop)

Check reachability **device to device**, which is easy to assume and wrong:

```bash
ssh comma@MAIN_IP 'ping -c2 -W2 GPS_IP'
```

### 1. Put both devices on the branch

```bash
for h in GPS_IP MAIN_IP; do
  ssh comma@$h 'cd /data/openpilot && git fetch origin gps-relay && git checkout -f -B gps-relay FETCH_HEAD'
done
```

If the working tree is dirty the checkout fails. Inspect before clobbering; `-f`
discards local edits.

### 2. Set the params

Two params, both `PERSISTENT`, so this survives reboots and is a one-time setup.

| param | type | meaning |
|---|---|---|
| `GpsPublish` | BOOL | offer this device's msgq over ZMQ |
| `GpsSource` | STRING | IP to take `gpsLocationExternal` from |

**GPS device** — publishes, sources nothing:

```bash
ssh comma@GPS_IP 'cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python -c "
from openpilot.common.params import Params
p = Params()
p.put_bool(\"GpsPublish\", True, block=True)
p.put(\"GpsSource\", \"\", block=True)"'
```

**Main device** — receives from the GPS device, *and also publishes*:

```bash
ssh comma@MAIN_IP 'cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python -c "
from openpilot.common.params import Params
p = Params()
p.put_bool(\"GpsPublish\", True, block=True)
p.put(\"GpsSource\", \"GPS_IP\", block=True)"'
```

`GpsPublish` on the **main** device is not a mistake. Receive mode only connects
outward and publishes nothing, so without it nothing on your laptop can subscribe to
the main device. The two modes coexist: receive connects, publish binds.

`PYTHONPATH` and `/usr/local/venv/bin/python` are both required. `/usr/bin/python3`
has no capnp.

### 3. Restart openpilot on both

Params are read live, but a branch change needs a restart:

```bash
for h in GPS_IP MAIN_IP; do ssh comma@$h 'sudo systemctl restart comma'; done
```

**Never `pkill manager.py` instead.** `launch_chffrplus.sh` does not loop — it falls
into `while true; do sleep 1; done` and openpilot stays dead until a real restart.

First boot after a code change runs `build.py`, which takes **several minutes**.
`manager.py` will be missing from the process list until it finishes. Wait it out.

### 4. Verify

```bash
for h in GPS_IP MAIN_IP; do
  echo "=== $h ==="
  ssh comma@$h '
    echo "GpsPublish  $(cat /data/params/d/GpsPublish 2>/dev/null)"
    echo "GpsSource   $(cat /data/params/d/GpsSource 2>/dev/null)"
    echo "gpsbridge   $(pgrep -f "[g]psbridge.gpsbridge" >/dev/null && echo yes || echo NO)"
    echo "bridges     $(pgrep -cx bridge)"
    ps aux | grep "[/]bridge" | awk "{print \"   \", \$12, \$13}"
    echo "ubloxd      $(pgrep -f "[u]bloxd.ubloxd" >/dev/null && echo yes || echo no)"'
done
```

Expected:

```
=== GPS_IP ===
GpsPublish  1
GpsSource
gpsbridge   yes
bridges     1
    bridge:                                  <- publish only
ubloxd      yes                              <- runs offroad because GpsPublish

=== MAIN_IP ===
GpsPublish  1
GpsSource   GPS_IP
gpsbridge   yes
bridges     2
    bridge: GPS_IP gpsLocationExternal       <- receive
    bridge:                                  <- publish
ubloxd      no                               <- correct; the bridge owns the endpoint
```

Note the `[u]bloxd` bracket. Without it `pgrep -f` matches its own command line and
always reports yes.

### 5. Confirm a real fix is crossing

On the **GPS device** first:

```bash
ssh comma@GPS_IP 'cd /data/openpilot && PYTHONPATH=/data/openpilot timeout 20 /usr/local/venv/bin/python -c "
from openpilot.cereal import messaging
import time
s = messaging.sub_sock(\"gpsLocationExternal\", timeout=1000)
for _ in range(40):
    m = messaging.recv_one_or_none(s)
    if m and m.gpsLocationExternal.hasFix:
        g = m.gpsLocationExternal
        print(f\"FIX {g.latitude:.6f},{g.longitude:.6f} acc={g.horizontalAccuracy:.1f} sats={g.satelliteCount}\"); break
    time.sleep(0.25)
else: print(\"no fix\")"'
```

Then the same command against `MAIN_IP`. Main has **no local GPS receiver**, so any
`gpsLocationExternal` there arrived over the relay — that is the proof.

Expect something like `FIX 32.751588,-117.196341 acc=1.8 sats=13`.

---

## Live display (optional)

A rerun window showing the position on a satellite map lives in a separate repo
(`comma_rerun`). Point its `HOST` constant at **MAIN_IP**, not the GPS device — the
whole point is seeing what the main device believes.

```bash
cd ~/openpilot && uv run python ~/comma_rerun/live_sub.py | python3 ~/comma_rerun/live_viz.py
```

Two interpreters on purpose: openpilot's venv is Python 3.12 and has cereal;
rerun-sdk is in the 3.14 user site. They cannot share an interpreter, so the two
halves are joined by a pipe.

If the viewer crashes with `transport error`, start it standalone first and let the
script connect to it:

```bash
rerun --port 9876 &
```

---

## Troubleshooting

Work down this list; it is ordered by how often each one actually happened.

### No fix anywhere, zero satellites

Check whether the receiver is *tracking* anything, which is different from having a
fix:

```bash
ssh comma@GPS_IP 'cd /data/openpilot && PYTHONPATH=/data/openpilot timeout 15 /usr/local/venv/bin/python -c "
from openpilot.cereal import messaging
import time
s = messaging.sub_sock(\"ubloxGnss\", timeout=1000)
sats = 0; rep = 0
t0 = time.monotonic()
while time.monotonic() - t0 < 10:
    m = messaging.recv_one_or_none(s)
    if m and m.ubloxGnss.which() == \"measurementReport\":
        rep += 1
        sats += len(m.ubloxGnss.measurementReport.measurements)
    time.sleep(0.02)
print(f\"reports={rep} satellites_tracked={sats}\")"'
```

`reports` high and `satellites_tracked=0` means the receiver is healthy and hears
nothing: no sky view, or the antenna is unseated. No amount of software helps. Take
the device outside and retry.

Cold start with no almanac takes a while even outdoors. `AssistNow` shortens it and
runs automatically via comma's AGPS proxy — look for `AssistNow messages sent` in
swaglog.

### Bridge is running, ports listening, no data

Two distinct causes, both seen repeatedly.

**Two bridges racing.** The second fails its port binds, prints `Failed to create ZMQ
publisher for [...]: Address already in use`, then *keeps running* holding sockets and
forwarding nothing. Ports all listening, zero data.

```bash
ssh comma@GPS_IP 'grep -c Failed /tmp/bridge.log'   # must be 0
```

**Stale msgq segment.** The bridge attaches its msgq subscriber the first time a ZMQ
client connects. If no publisher ever existed for that endpoint it lands on a dead
shared-memory segment and forwards nothing, forever. This happens when the bridge
starts before `ubloxd`.

Both are fixed the same way — kill the bridge and let `gpsbridge` restart it, after
confirming `ubloxd` is up:

```bash
ssh comma@GPS_IP 'pkill -9 -x bridge'    # gpsbridge restarts it within ~2s
```

The current `gpsbridge` guards against this by waiting for real local
`gpsLocationExternal` data before starting the publish bridge, but the guard only
applies on a device that sources its own GPS.

### `ubloxd` not running on the GPS device

It must run even though the device is offroad. The gate is:

```python
(started or params.get_bool("GpsPublish")) and use_ublox and not gps_relayed(params)
```

If `GpsPublish` is unset, `ubloxd` only runs onroad and the GPS device produces
nothing while parked. Set the param and restart with `sudo systemctl restart comma`.

On an older revision without this gate, start them by hand:

```bash
ssh comma@GPS_IP 'cd /data/openpilot && export PYTHONPATH=/data/openpilot
(setsid /usr/local/venv/bin/python -m openpilot.system.ubloxd.pigeond >/tmp/pigeond.log 2>&1 </dev/null &)
(setsid /usr/local/venv/bin/python -m openpilot.system.ubloxd.ubloxd  >/tmp/ubloxd.log  2>&1 </dev/null &)'
```

### `ubloxd` running on the **main** device

It will fight the receive bridge for the `gpsLocationExternal` endpoint — msgq allows
exactly one publisher. Means `GpsSource` is unset or empty. Re-set it and restart.

### Laptop cannot subscribe to the main device

`GpsPublish` is not set on main. Receive mode publishes nothing outward. Set it.

### Anything launched over SSH dies immediately

A plain `&` background job dies when the ssh channel closes. Always:

```bash
ssh comma@IP 'cd /data/openpilot && (setsid ./some_command >/tmp/log 2>&1 </dev/null &) ; exit 0'
```

Also: a foreground `ssh ... 'cmd &; sleep N'` will hang until the timeout, because ssh
waits for every inherited fd.

### `pgrep` reports processes that are not running

`pgrep -f foo` matches the shell running the `pgrep`. Bracket the first character:
`pgrep -f "[f]oo"`. Same trap with `pkill -f` — it kills its own parent shell, which
shows up as exit code 143 or 144.

### Params vanish after a reboot or ignition cycle

`GpsPublish` and `GpsSource` are `PERSISTENT` and survive.

`ParkingDestination` and `ParkingNavEnabled` are **not** — they are
`CLEAR_ON_MANAGER_START | CLEAR_ON_OFFROAD_TRANSITION`, so every reboot and every
onroad/offroad transition wipes them. Re-set after each:

```bash
ssh comma@MAIN_IP 'cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python -c "
from openpilot.common.params import Params
p = Params()
p.put(\"ParkingDestination\", {\"latitude\": LAT, \"longitude\": LON}, block=True)
p.put_bool(\"ParkingNavEnabled\", True, block=True)"'
```

Pass a **dict**, not a JSON string: the conversion table has `(dict, JSON)` but not
`(str, JSON)`, and a string raises `TypeError`. Likewise `put_bool` for BOOL params —
`put` with a string raises, because only `(bool, BOOL)` exists.

---

## How it works

### The pipeline

Normally:

```
u-blox HW -> pigeond -> ubloxRaw -> ubloxd -> gpsLocationExternal -> parknavd, UI, hardwared
```

The relay replaces the final publisher on the main device:

```
GPS device:   u-blox -> pigeond -> ubloxd -> gpsLocationExternal -> bridge (publish)
                                                                      |  ZMQ/TCP
main device:  bridge (receive) -> gpsLocationExternal (local msgq) ----+
                                        |
                                        +-> parknavd, UI, hardwared
              bridge (publish) -> ZMQ -> your laptop
```

### The splice point

`cereal/messaging/bridge` already relays both directions; `bridge.cc` picks the mode
by argument count:

```cpp
bool is_zmq_to_msgq = argc > 2;
```

- `bridge` — msgq to ZMQ, every service. Binds one port per service.
- `bridge <ip> <whitelist>` — ZMQ to msgq, whitelisted only. Connects outward.

Ports are an FNV-1a hash of the service name into `[8023, 65535]`, computed
identically on both ends, so there is no port to configure.

### The pieces added

**`openpilot/system/gpsbridge/gpsbridge.py`** — a manager process that reads the two
params every 2s and runs `bridge` in whichever modes are called for, restarting a
child that exits. Its gate deliberately ignores `started`, so the relay stays up
while parked.

**`process_config.py`** — the `ublox` gate gained `or GpsPublish` (run offroad on a
GPS device) and `and not gps_relayed(params)` (stay off on a receiving device, so the
bridge owns the endpoint).

**`params_keys.h`** — `GpsPublish` and `GpsSource`, both `PERSISTENT`.

### Known limitations

- **Position offset** equals the physical separation of the two devices. Sub-metre if
  both are in the same car, well inside the ~2–4 m noise floor.
- **`logMonoTime` is rewritten** by the republisher, so it is the main device's clock.
  The payload's own `unixTimestampMillis` is untouched. Wifi adds a few tens of ms.
- **No authentication.** Anything on the network can subscribe to a publishing
  device, or feed a receiving one.
- Only `gpsLocationExternal` is whitelisted. `ubloxGnss` (raw satellite data) stays
  local to the GPS device.
