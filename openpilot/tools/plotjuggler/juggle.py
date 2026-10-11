#!/usr/bin/env python3
"""
juggle.py — redirects to cabana, which now includes PlotJuggler's functionality.

PlotJuggler has been merged into cabana. This script translates juggle.py arguments
to their cabana equivalents and launches cabana.
"""
import os
import sys
import shutil
import subprocess
import argparse

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CABANA_BIN = os.path.join(BASEDIR, "openpilot", "tools", "cabana", "cabana")


def print_redirect_banner():
  bold  = "\033[1m"   if sys.stdout.isatty() else ""
  cyan  = "\033[96m"  if sys.stdout.isatty() else ""
  reset = "\033[0m"   if sys.stdout.isatty() else ""
  print(f"{bold}{cyan}╔══════════════════════════════════════════════════════╗{reset}")
  print(f"{bold}{cyan}║{reset}  PlotJuggler has moved into cabana                   {bold}{cyan}║{reset}")
  print(f"{bold}{cyan}║{reset}                                                      {bold}{cyan}║{reset}")
  print(f"{bold}{cyan}║{reset}  Use cabana directly:                                {bold}{cyan}║{reset}")
  print(f"{bold}{cyan}║{reset}    tools/cabana/cabana --demo                        {bold}{cyan}║{reset}")
  print(f"{bold}{cyan}║{reset}    tools/cabana/cabana --demo --layout tuning        {bold}{cyan}║{reset}")
  print(f"{bold}{cyan}║{reset}    tools/cabana/cabana <route>                       {bold}{cyan}║{reset}")
  print(f"{bold}{cyan}║{reset}                                                      {bold}{cyan}║{reset}")
  print(f"{bold}{cyan}║{reset}  This script forwards your args to cabana for now.   {bold}{cyan}║{reset}")
  print(f"{bold}{cyan}╚══════════════════════════════════════════════════════╝{reset}")
  print()


def build_cabana_cmd(args) -> list[str]:
  """Map juggle.py args to cabana args."""
  cmd = [CABANA_BIN]

  if args.demo:
    cmd.append("--demo")

  if args.stream:
    cmd.append("--msgq")

  if args.dbc:
    cmd += ["--dbc", args.dbc]

  # --layout: pass by name (cabana resolves built-in names automatically)
  if args.layout:
    layout = args.layout
    # strip .xml extension if someone passes the old plotjuggler layout path
    if layout.endswith(".xml"):
      layout = os.path.splitext(os.path.basename(layout))[0]
    cmd += ["--layout", layout]

  if args.route_or_segment_name:
    cmd.append(args.route_or_segment_name.strip())

  return cmd


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Redirects to cabana, which now includes PlotJuggler functionality.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--demo",    action="store_true", help="Use the demo route")
  parser.add_argument("--stream",  action="store_true", help="Live streaming mode (--msgq in cabana)")
  parser.add_argument("--layout",  nargs="?",           help="Load a pre-defined layout by name or path")
  parser.add_argument("--dbc",                          help="DBC file to load for CAN parsing")
  # kept for backward compat but ignored — cabana always parses CAN
  parser.add_argument("--can",         action="store_true", help="(ignored — cabana always parses CAN)")
  parser.add_argument("--no-migration",action="store_true", help="(ignored — cabana has no migration step)")
  parser.add_argument("--install",     action="store_true", help="(ignored — cabana needs no install)")
  parser.add_argument("route_or_segment_name", nargs="?",   help="Route or segment to load")

  if len(sys.argv) == 1:
    print_redirect_banner()
    parser.print_help()
    sys.exit(0)

  args = parser.parse_args()
  print_redirect_banner()

  if args.install:
    print("cabana is built from source — run: scons openpilot/tools/cabana/cabana")
    sys.exit(0)

  if not os.path.exists(CABANA_BIN):
    print(f"cabana binary not found at: {CABANA_BIN}")
    print("Build it first:  scons openpilot/tools/cabana/cabana")
    sys.exit(1)

  cmd = build_cabana_cmd(args)
  print(f"Running: {' '.join(cmd)}\n")
  sys.exit(subprocess.call(cmd))
