#!/usr/bin/env python3
import csv

import click

from selfdrive.boardd.boardd import can_capnp_to_can_list
from tools.plotjuggler.juggle import load_segment


@click.command()
@click.argument("rlog_filename")
@click.option("--output", default="can_output.csv", help="Output CAN csv filename")
def rlog_to_can_csv(rlog_filename, output):
  """This is a script to extract CAN messages from rlog to csv file."""
  lr = load_segment(rlog_filename)

  with open(output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Bus", "MessageID", "Message", "MessageLength", "Time"])

    start_time = None
    for m in lr:
      if m.which() == "can":
        cl = can_capnp_to_can_list(m.can)
        for can in cl:
          if start_time is None:
            start_time = m.logMonoTime

          address, _, msg, src = can
          writer.writerow(
            [
              str(src),
              str(hex(address)),
              f"0x{msg.hex()}",
              len(msg),
              str((m.logMonoTime - start_time) / 1e9),
            ]
          )


if __name__ == "__main__":
  rlog_to_can_csv() # pylint: disable=E1120
