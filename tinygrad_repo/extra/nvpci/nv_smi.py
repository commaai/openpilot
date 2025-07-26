#!/usr/bin/env python3

from tinygrad.runtime.support.system import System
import argparse, glob, os, re, time, subprocess, sys

def scan_devs_based_on_lock(prefix:str) -> list[str]:
  devs = []
  for dev in glob.glob(f'/tmp/{prefix}_*.lock'):
    dev_id = dev[8:-5]
    if os.path.exists(f"/sys/bus/pci/devices/{dev_id}"): devs.append(dev_id)
  return devs

def _do_reset_device(pci_bus): System.pci_reset(pci_bus)
def _is_module_loaded(name: str) -> bool: return os.path.isdir(f"/sys/module/{name}")

def cmd_remove_module(args):
  to_unload = [m for m in ["nvidia_drm", "nvidia_modeset", "nvidia_uvm", "nvidia"] if _is_module_loaded(m)]
  if not to_unload:
    print("NVIDIA kernel modules are not loaded")
  else:
    print("Removing NVIDIA kernel modules:", ", ".join(to_unload))
    try: subprocess.run(["sudo", "modprobe", "-r", *to_unload], check=True)
    except subprocess.CalledProcessError as e:
      print("Failed to unload all modules — they may be in use.", file=sys.stderr)
      sys.exit(e.returncode)

def cmd_insert_module(args):
  cmd_remove_module(args)
  cmd_reset_devices(args)

  if not os.path.exists("/sys/module/nvidia"):
    print("Inserting nvidia kernel module")
    subprocess.run(["nvidia-smi"], check=True)
  else: print("Nvidia kernel module already loaded")

def cmd_reset_devices(args):
  devs = scan_devs_based_on_lock("nv")
  dev_to_reset = args.pci_bus if 'pci_bus' in args.__dir__() else ""

  for dev in devs:
    if dev.startswith(dev_to_reset):
      print(f"Resetting device {dev}")
      _do_reset_device(dev)
      time.sleep(0.2)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(required=True, dest="cmd")

  parser_insmod = subparsers.add_parser('insmod', help='Insert a nvidia kernel module')
  parser_insmod.set_defaults(func=cmd_insert_module)

  parser_rmmod = subparsers.add_parser('rmmod', help='Remove a nvidia kernel module')
  parser_rmmod.set_defaults(func=cmd_remove_module)

  parser_reset = subparsers.add_parser('reset', help='Reset a nvidia device')
  parser_reset.add_argument('--pci_bus', type=str, default="", help='PCI bus ID of the device to reset')
  parser_reset.set_defaults(func=cmd_reset_devices)

  args = parser.parse_args()
  if args.cmd is None:
    parser.print_help(sys.stderr)
    sys.exit(1)

  args.func(args)
