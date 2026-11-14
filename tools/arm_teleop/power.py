#!/usr/bin/env python3
import glob
import grp
import os
from pathlib import Path
import subprocess
import sys
import time


def run(*args):
  return subprocess.run(args, text=True, capture_output=True)


def write(path, value):
  try:
    Path(path).write_text(value)
  except OSError as e:
    print(f"{path}: {e}", flush=True)


def hotplug():
  for pattern in ("/sys/bus/usb/devices/*/power/control", "/sys/bus/platform/devices/*.ssusb/power/control"):
    for path in glob.glob(pattern):
      if Path(path).read_text().strip() != "on":
        write(path, "on")
  for path in glob.glob("/dev/ttyACM*"):
    os.chown(path, 0, grp.getgrnam("dialout").gr_gid)
    os.chmod(path, 0o660)


def install_gateway_key():
  source = Path("/data/arm_teleop_gateway.pub")
  if not source.exists():
    return
  kind, key, *_ = source.read_text().split()
  if kind != "ssh-ed25519":
    raise ValueError("Expected an Ed25519 gateway public key")
  target = Path("/tmp/authorized_keys")
  lines = target.read_text().splitlines() if target.exists() else []
  lines = [line for line in lines if not line.endswith(" mici-game-gateway")]
  lines.append(f'restrict,port-forwarding,permitopen="127.0.0.1:8022",command="/bin/false" {kind} {key} mici-game-gateway')
  target.write_text("\n".join(lines) + "\n")
  target.chmod(0o644)


def main():
  assert os.geteuid() == 0
  install_gateway_key()
  result = run("systemctl", "mask", "--runtime", "--now", "power_monitor.service", "power_drop_monitor.service",
               "sleep.target", "suspend.target", "hibernate.target", "hybrid-sleep.target")
  print(result.stderr, end="")
  run("systemctl", "daemon-reload")
  write("/sys/module/usbcore/parameters/autosuspend", "-1")
  hotplug()

  for path in glob.glob("/sys/devices/system/cpu/cpu[0-7]/online"):
    write(path, "1")
  for path in glob.glob("/sys/devices/system/cpu/cpufreq/policy*/scaling_governor"):
    write(path, "performance")
  for path in glob.glob("/sys/devices/system/cpu/cpu[0-7]/cpuidle/state*/disable"):
    write(path, "1")

  conf = Path("/run/NetworkManager/conf.d/99-arm-teleop.conf")
  conf.parent.mkdir(parents=True, exist_ok=True)
  conf.write_text("[device]\nwifi.scan-rand-mac-address=no\n\n[connection]\nwifi.powersave=2\n" +
                  "wifi.cloned-mac-address=permanent\nethernet.cloned-mac-address=permanent\n")
  connections = run("nmcli", "-t", "-f", "UUID,TYPE", "connection", "show")
  for line in connections.stdout.splitlines():
    uuid, kind = line.split(":", 1)
    if kind == "802-11-wireless":
      result = run("nmcli", "connection", "modify", uuid, "802-11-wireless.powersave", "2",
                   "802-11-wireless.mac-address-randomization", "1", "802-11-wireless.cloned-mac-address", "permanent")
      print(result.stderr, end="")
    elif kind == "802-3-ethernet":
      run("nmcli", "connection", "modify", uuid, "802-3-ethernet.cloned-mac-address", "permanent")
  run("nmcli", "general", "reload", "conf")
  result = run("nmcli", "device", "reapply", "wlan0")
  print(result.stdout, result.stderr, end="")
  print("Arm teleop: performance mode, USB always on, idle shutdown disabled, permanent MACs", flush=True)


if __name__ == "__main__":
  if "--watch" in sys.argv:
    while True:
      hotplug()
      time.sleep(2)
  else:
    main()
