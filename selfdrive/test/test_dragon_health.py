#!/usr/bin/env python3
"""Dragon Q6A system health check.

Checks connectivity (NCM/wifi/bluetooth), openpilot processes, camera FPS,
captures sample images, and runs model replay.

Invoked from the host via:  dragon.py health
"""
import os
import subprocess
import sys
import time
from pathlib import Path

CAMERA_SERVICES = ('roadCameraState', 'wideRoadCameraState')
EXPECTED_FRAME_INTERVAL_MS = 50.0
FRAME_INTERVAL_TOLERANCE_MS = 1.0
SNAPSHOT_DIR = "/tmp/dragon_health"


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

def run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1, "", ""


def check_ncm():
    section("NCM (USB networking)")
    _, out, _ = run(["ip", "-4", "-o", "addr", "show"])
    for line in out.splitlines():
        if "usb" in line or "192.168.42." in line:
            ok(line.strip())
            return True
    warn("No NCM interface with 192.168.42.x found")
    return False


def check_wifi():
    section("WiFi")
    code, out, _ = run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"])
    if code != 0:
        code2, out2, _ = run(["ip", "link", "show", "wlan0"])
        if code2 == 0 and "UP" in out2:
            ok("wlan0 is UP (nmcli unavailable)")
            return True
        warn("Cannot query wifi")
        return False

    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) >= 3 and parts[1] == "wifi" and parts[2] == "connected":
            conn = parts[3] if len(parts) > 3 else ""
            ok(f"WiFi connected: {conn}")
            return True

    warn("WiFi not connected")
    return False


def check_bluetooth():
    section("Bluetooth")
    code, out, _ = run(["bluetoothctl", "show"], timeout=5)
    if code == 0 and out:
        powered = "Powered: yes" in out
        (ok if powered else warn)("Bluetooth " + ("powered on" if powered else "not powered"))
        return powered

    code, out, _ = run(["hciconfig", "hci0"])
    if code == 0:
        up = "UP RUNNING" in out
        (ok if up else warn)("hci0 " + ("UP RUNNING" if up else "down"))
        return up

    warn("No bluetooth tools available")
    return False


def check_processes():
    section("Openpilot Processes")
    results = {}
    try:
        import cereal.messaging as messaging
        sm = messaging.SubMaster(['managerState'])
        for _ in range(20):
            sm.update(200)
            if sm.updated['managerState']:
                break
        if sm.updated['managerState']:
            for p in sm['managerState'].processes:
                results[p.name] = {
                    'running': p.running,
                    'shouldBeRunning': p.shouldBeRunning,
                    'pid': p.pid,
                    'exitCode': p.exitCode,
                }
        else:
            warn("managerState not received — is manager running?")
    except Exception as e:
        warn(f"Cannot read managerState: {e}")

    if not results:
        warn("No process state available")
        return False

    all_ok = True
    for proc, info in sorted(results.items()):
        info = results[proc]
        if info['running']:
            ok(f"{proc:25s}  pid={info['pid']}")
        elif info['shouldBeRunning']:
            fail(f"{proc:25s}  SHOULD BE RUNNING (exit={info.get('exitCode', '?')})")
            all_ok = False
    return all_ok


def measure_fps(duration=5):
    section(f"Camera FPS ({duration}s sample)")
    try:
        import cereal.messaging as messaging
        import numpy as np
    except ImportError:
        fail("cereal.messaging not available")
        return {}

    socks = {cam: messaging.sub_sock(cam, conflate=False, timeout=100) for cam in CAMERA_SERVICES}
    time.sleep(0.2)
    for s in socks.values():
        messaging.drain_sock(s)

    frame_ids = {cam: [] for cam in CAMERA_SERVICES}
    frame_times = {cam: [] for cam in CAMERA_SERVICES}
    start = time.monotonic()
    while time.monotonic() - start < duration:
        for cam, sock in socks.items():
            for msg in messaging.drain_sock(sock):
                frame = getattr(msg, cam)
                frame_ids[cam].append(frame.frameId)
                frame_times[cam].append(frame.timestampSof)
        time.sleep(0.02)

    fps_results = {}
    for cam in CAMERA_SERVICES:
        ids = frame_ids[cam]
        times = frame_times[cam]
        n = len(ids)
        skips = int(np.sum(np.diff(ids) > 1)) if n > 1 else 0

        if n == 0:
            fail(f"{cam:25s}  no frames received")
            fps_results[cam] = {'fps': 0, 'mean_ms': 0, 'ok': False}
        elif len(times) < 2:
            fail(f"{cam:25s}  only one timestamp received")
            fps_results[cam] = {'fps': 0, 'mean_ms': 0, 'ok': False}
        else:
            intervals_ms = np.diff(times) / 1e6
            mean_ms = float(np.mean(intervals_ms))
            fps = 1000.0 / mean_ms if mean_ms > 0 else 0
            min_ms = float(np.min(intervals_ms))
            max_ms = float(np.max(intervals_ms))
            cadence_ok = abs(mean_ms - EXPECTED_FRAME_INTERVAL_MS) <= FRAME_INTERVAL_TOLERANCE_MS
            fps_results[cam] = {'fps': fps, 'mean_ms': mean_ms, 'ok': cadence_ok}
            msg = (f"{cam:25s}  {fps:.2f} fps, {mean_ms:.2f} ms/frame "
                   f"({n} frames, {skips} skips, range {min_ms:.2f}-{max_ms:.2f} ms, "
                   f"expected {EXPECTED_FRAME_INTERVAL_MS:.0f}±{FRAME_INTERVAL_TOLERANCE_MS:.0f} ms)")
            (ok if cadence_ok else fail)(msg)
    return fps_results


def image_has_color(img):
    if len(img.shape) != 3 or img.shape[2] < 3:
        return False
    # JPEG snapshots should not be monochrome: require visible RGB channel spread.
    channel_delta = max(
        float(abs(img[:, :, 0].astype("int16") - img[:, :, 1].astype("int16")).mean()),
        float(abs(img[:, :, 0].astype("int16") - img[:, :, 2].astype("int16")).mean()),
        float(abs(img[:, :, 1].astype("int16") - img[:, :, 2].astype("int16")).mean()),
    )
    return channel_delta >= 2.0


def boost_saturation(img, factor=4.0):
    rgb = img.astype("float32")
    luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])[:, :, None]
    return (luma + (rgb - luma) * factor).clip(0, 255).astype("uint8")


def capture_snapshots():
    section("Camera Snapshots")
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    try:
        from msgq.visionipc import VisionIpcClient, VisionStreamType
        from openpilot.system.camerad.snapshot import extract_image, jpeg_write
    except ImportError as e:
        fail(f"Cannot import snapshot deps: {e}")
        return False

    streams = {
        'road': VisionStreamType.VISION_STREAM_ROAD,
        'wide_road': VisionStreamType.VISION_STREAM_WIDE_ROAD,
    }
    all_ok = True
    for name, stream_type in streams.items():
        try:
            client = VisionIpcClient("camerad", stream_type, True)
            if not client.connect(True):
                fail(f"{name}: could not connect to VisionIPC")
                all_ok = False
                continue
            buf = client.recv()
            if buf is None or len(buf.data) == 0:
                fail(f"{name}: empty frame")
                all_ok = False
                continue
            img = boost_saturation(extract_image(buf))
            path = os.path.join(SNAPSHOT_DIR, f"{name}.jpg")
            jpeg_write(path, img)
            if image_has_color(img):
                ok(f"{name}: {img.shape[1]}x{img.shape[0]} color saved to {path}")
            else:
                fail(f"{name}: {img.shape[1]}x{img.shape[0]} saved to {path}, but image is monochrome")
                all_ok = False
        except Exception as e:
            fail(f"{name}: {e}")
            all_ok = False
    return all_ok


def run_model_replay():
    section("Model Replay")
    try:
        import matplotlib
    except ImportError:
        print("  Installing matplotlib...")
        code, _, err = run(["sudo", sys.executable, "-m", "pip", "install",
                            "--break-system-packages", "matplotlib"], timeout=120)
        if code != 0:
            fail(f"pip install matplotlib failed: {err}")
            return False
        ok("matplotlib installed")

    replay_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "process_replay", "model_replay.py")
    if not os.path.isfile(replay_script):
        fail(f"model_replay.py not found at {replay_script}")
        return False

    print("  Running model_replay.py (this may take a while)...")
    env = os.environ.copy()
    env["COMMA_CACHE"] = "/data/comma_download_cache"
    proc = subprocess.Popen([sys.executable, replay_script],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    for line in proc.stdout:
        print(f"  {line.rstrip()}")
    proc.wait()

    if proc.returncode == 0:
        ok("Model replay passed")
        return True
    fail(f"Model replay failed (exit={proc.returncode})")
    return False


def system_info():
    section("System Info")
    code, out, _ = run(["uname", "-r"])
    if code == 0:
        ok(f"Kernel: {out}")

    code, out, _ = run(["uptime", "-p"])
    if code == 0:
        ok(f"Uptime: {out}")

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    ok(line.strip())
    except Exception:
        pass

    try:
        temps = []
        for tz in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            try:
                t = int((tz / "temp").read_text().strip()) / 1000
                name = (tz / "type").read_text().strip()
                temps.append(f"{name}={t:.0f}C")
            except Exception:
                continue
        if temps:
            ok(f"Temps: {', '.join(temps[:6])}")
    except Exception:
        pass

    code, out, _ = run(["df", "-h", "/data"])
    if code == 0 and len(out.splitlines()) > 1:
        ok(f"Disk: {out.splitlines()[-1].strip()}")


def wait_for_openpilot(timeout=600):
    section("Waiting for openpilot")
    try:
        import cereal.messaging as messaging
    except ImportError:
        warn("cereal not importable, proceeding anyway")
        return True

    sock = messaging.sub_sock('managerState', timeout=1000)
    start = time.monotonic()
    last_print = 0
    while time.monotonic() - start < timeout:
        if messaging.drain_sock(sock):
            ok(f"openpilot is up ({time.monotonic() - start:.0f}s)")
            return True
        now = time.monotonic()
        if now - last_print >= 10:
            elapsed = now - start
            compiling = run(["pgrep", "-f", "scons"])[0] == 0
            print(f"  ... {'compiling' if compiling else 'waiting for manager'} ({elapsed:.0f}s)")
            last_print = now
    fail(f"openpilot did not start within {timeout}s")
    return False


def main():
    print(f"Dragon Q6A Health Check — {time.strftime('%Y-%m-%d %H:%M:%S')}")

    system_info()
    ready = wait_for_openpilot()
    if not ready:
        print("\n  Proceeding with checks anyway...\n")

    results = {
        'ncm':          check_ncm(),
        'wifi':         check_wifi(),
        'bluetooth':    check_bluetooth(),
        'processes':    check_processes(),
        'fps':          measure_fps(),
        'snapshots':    capture_snapshots(),
        'model_replay': run_model_replay(),
    }

    section("Summary")
    fps = results.get('fps', {})
    fps_ok = all(v['ok'] for v in fps.values()) if fps else False
    checks = [
        ("NCM",          results['ncm']),
        ("WiFi",         results['wifi']),
        ("Bluetooth",    results['bluetooth']),
        ("Processes",    results['processes']),
        ("Snapshots",    results['snapshots']),
        ("Model Replay", results['model_replay']),
        ("Camera FPS",   fps_ok),
    ]

    all_pass = True
    for name, passed in checks:
        (ok if passed else fail)(name)
        all_pass = all_pass and passed

    if fps:
        fps_str = ", ".join(f"{c.replace('CameraState','')}={v['fps']:.2f}fps/{v['mean_ms']:.2f}ms" for c, v in fps.items())
        print(f"\n  FPS: {fps_str}")
    print(f"  Snapshots: {SNAPSHOT_DIR}")
    print()
    print("  ALL CHECKS PASSED" if all_pass else "  SOME CHECKS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
