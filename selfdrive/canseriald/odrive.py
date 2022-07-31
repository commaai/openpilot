#!/usr/bin/env python

import asyncio
from typing import List

import cantools
import can
from can.notifier import MessageRecipient


db = cantools.database.load_file('../../odrivebody/odrive_comma_body.dbc')

odrive0_node_id = 0
odrive1_node_id = 1


def get_arbitration_id(node_id: int, command_id: int) -> int:
  return (node_id << 5) + command_id


def get_command_id(arbitration_id: int) -> int:
  return arbitration_id & 0b00011111


def print_message(message: can.Message) -> None:
  command_id = get_command_id(message.arbitration_id)
  encoded_message = db.decode_message(command_id, message.data)
  print(message)
  print(encoded_message)


def request_message(bus: can.interface.Bus, node_id: int, command_id: int) -> None:
  message = can.Message(arbitration_id=get_arbitration_id(node_id, command_id),
                        is_remote_frame=True, is_extended_id=False)
  bus.send(message)


def is_heartbeat_message(message: can.Message) -> bool:
  return get_command_id(message.arbitration_id) == 1


async def main() -> None:

  # pylint: disable=abstract-class-instantiated
  bus_interface = can.interface.Bus(bustype='slcan', channel='/dev/ttyACM0', bitrate=250000)

  with bus_interface as bus:

    listeners: List[MessageRecipient] = [
        lambda msg: print_message(msg) if not is_heartbeat_message(msg) else None,
    ]

    loop = asyncio.get_running_loop()
    can.Notifier(bus, listeners, loop=loop)

    while(True):
      request_message(bus, odrive0_node_id, 23)
      await asyncio.sleep(1)

    # Clean-up
    # notifier.stop()


if __name__ == "__main__":
  asyncio.run(main())
