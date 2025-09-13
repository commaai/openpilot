#!/usr/bin/env python3
import os
import sys
import argparse
import re

from openpilot.common.basedir import BASEDIR
from openpilot.tools.lib.auth_config import get_token
from openpilot.tools.lib.api import CommaApi

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper for connecting to devices over the comma prime SSH proxy.\
      Adding your SSH key to your SSH config is recommended for more convenient use; see https://docs.comma.ai/how-to/connect-to-comma/.")
  parser.add_argument("device", help="device name or dongle id")
  parser.add_argument("--host", help="ssh jump server host", default="ssh.comma.ai")
  parser.add_argument("--port", help="ssh jump server port", default=22, type=int)
  parser.add_argument("--key", help="ssh key", default=os.path.join(BASEDIR, "system/hardware/tici/id_rsa"))
  parser.add_argument("--debug", help="enable debug output", action="store_true")
  args = parser.parse_args()

  r = CommaApi(get_token()).get("v1/me/devices")
  devices = {x['dongle_id']: x['alias'] for x in r}

  if not re.match("[0-9a-zA-Z]{16}", args.device):
    user_input = args.device.replace(" ", "").lower()
    matches = { k: v for k, v in devices.items() if isinstance(v, str) and user_input in v.replace(" ", "").lower() }
    if len(matches) == 1:
      dongle_id = list(matches.keys())[0]
    else:
      print(f"failed to look up dongle id for \"{args.device}\"", file=sys.stderr)
      if len(matches) > 1:
        print("found multiple matches:", file=sys.stderr)
        for k, v in matches.items():
          print(f"  \"{v}\" ({k})", file=sys.stderr)
      exit(1)
  else:
    dongle_id = args.device

  name = dongle_id
  if dongle_id in devices:
    name = f"{devices[dongle_id]} ({dongle_id})"
  print(f"connecting to {name} through {args.host}:{args.port} ...")

  command = [
    "ssh",
    "-i", args.key,
    "-o", f"ProxyCommand=ssh -i {args.key} -W %h:%p -p %p %h@{args.host}",
    "-p", str(args.port),
  ]
  if args.debug:
    command += ["-v"]
  command += [
    f"comma@{dongle_id}",
  ]
  if args.debug:
    print(" ".join([f"'{c}'" if " " in c else c for c in command]))
  os.execvp(command[0], command)
