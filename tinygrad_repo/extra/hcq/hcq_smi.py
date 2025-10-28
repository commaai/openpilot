#!/usr/bin/env python3

from tinygrad.runtime.support.system import System
import argparse, glob, os, re, time, subprocess, sys

def scan_devs_based_on_lock(prefix:str, args) -> list[str]:
  target_dev = args.pci_bus if 'pci_bus' in args.__dir__() else ""

  devs = []
  for dev in glob.glob(f'/tmp/{prefix}_*.lock'):
    dev_id = dev[8:-5]
    if os.path.exists(f"/sys/bus/pci/devices/{dev_id}") and dev_id.startswith(target_dev): devs.append(dev_id)
  return devs

def _do_reset_device(pci_bus): System.pci_reset(pci_bus)
def _is_module_loaded(name: str) -> bool: return os.path.isdir(f"/sys/module/{name}")

def cmd_remove_module(args):
  modules = ["nvidia_drm", "nvidia_modeset", "nvidia_uvm", "nvidia", "ast"] if args.backend == "nv" else ["amdgpu"]
  to_unload = [m for m in modules if _is_module_loaded(m)]
  if not to_unload: print("Kernel modules are not loaded")
  else:
    print("Removing kernel modules:", ", ".join(to_unload))
    try: subprocess.run(["sudo", "modprobe", "-r", *to_unload], check=True)
    except subprocess.CalledProcessError as e:
      print("Failed to unload all modules — they may be in use.", file=sys.stderr)
      sys.exit(e.returncode)

def cmd_insert_module(args):
  cmd_remove_module(args)
  cmd_reset_devices(args)

  module = "nvidia" if args.backend == "nv" else "amdgpu"
  if _is_module_loaded(module):
    print(f"{module} kernel module already loaded")
    return

  print(f"Inserting kernel module: {module}")
  if args.backend == "nv":
    subprocess.run(["nvidia-smi"], check=True)
  elif args.backend == "amd":
    subprocess.run(["sudo", "modprobe", "amdgpu"], check=True)

def cmd_reset_devices(args):
  devs = scan_devs_based_on_lock({"amd":"am", "nv":"nv"}[args.backend], args)

  for dev in devs:
    print(f"Resetting device {dev}")
    if args.backend != "amd": _do_reset_device(dev)
    time.sleep(0.2)

def cmd_show_pids(args):
  devs = scan_devs_based_on_lock(prefix:={"amd":"am", "nv":"nv"}[args.backend], args)

  for dev in devs:
    try:
      pid = subprocess.check_output(['sudo', 'lsof', f'/tmp/{prefix}_{dev}.lock']).decode('utf-8').strip().split('\n')[1].split()[1]
      print(f"{dev}: {pid}")
    except subprocess.CalledProcessError: print(f"{dev}: No processes found using this device")

def cmd_kill_pids(args):
  devs = scan_devs_based_on_lock(prefix:={"amd":"am", "nv":"nv"}[args.backend], args)

  for dev in devs:
    try:
      pid = subprocess.check_output(['sudo', 'lsof', f'/tmp/{prefix}_{dev}.lock']).decode('utf-8').strip().split('\n')[1].split()[1]
      print(f"{dev}: {pid}")
    except subprocess.CalledProcessError: print(f"{dev}: No processes found using this device")

def cmd_kill_pids(args):
  devs = scan_devs_based_on_lock(prefix:={"amd":"am", "nv":"nv"}[args.backend], args)

  for dev in devs:
    for i in range(128):
      if i > 0: time.sleep(0.2)

      try:
        try: pid = subprocess.check_output(['sudo', 'lsof', f'/tmp/{prefix}_{dev}.lock']).decode('utf-8').strip().split('\n')[1].split()[1]
        except subprocess.CalledProcessError: break

        print(f"Killing process {pid} (which uses {dev})")
        subprocess.run(['sudo', 'kill', '-9', pid], check=True)
      except subprocess.CalledProcessError as e:
        print(f"Failed to kill process for device {dev}: {e}", file=sys.stderr)

def add_common_commands(parent_subparsers):
  p_insmod = parent_subparsers.add_parser("insmod", help="Insert a kernel module")
  p_insmod.set_defaults(func=cmd_insert_module)

  p_rmmod = parent_subparsers.add_parser("rmmod", help="Remove a kernel module")
  p_rmmod.set_defaults(func=cmd_remove_module)

  p_reset = parent_subparsers.add_parser("reset", help="Reset a device")
  p_reset.add_argument("--pci_bus", default="", help="PCI bus ID of the device to reset")
  p_reset.set_defaults(func=cmd_reset_devices)

  p_reset = parent_subparsers.add_parser("pids", help="Show pids of processes using the device")
  p_reset.add_argument("--pci_bus", default="", help="PCI bus ID of the device")
  p_reset.set_defaults(func=cmd_show_pids)

  p_reset = parent_subparsers.add_parser("kill_pids", help="Kill pids of processes using the device")
  p_reset.add_argument("--pci_bus", default="", help="PCI bus ID of the device")
  p_reset.set_defaults(func=cmd_kill_pids)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  backend_subparsers = parser.add_subparsers(dest="backend", required=True, metavar="{nv,amd}", help="Hardware backend to target")

  nv_parser = backend_subparsers.add_parser("nv", help="NVIDIA GPUs")
  nv_commands = nv_parser.add_subparsers(dest="command", required=True)
  add_common_commands(nv_commands)

  amd_parser = backend_subparsers.add_parser("amd", help="AMD GPUs")
  amd_commands = amd_parser.add_subparsers(dest="command", required=True)
  add_common_commands(amd_commands)

  args = parser.parse_args()
  if args.command is None:
    parser.print_help(sys.stderr)
    sys.exit(1)

  args.func(args)
