#!/usr/bin/env python3
import asyncio

import cantools
import can

import cereal.messaging as messaging
from selfdrive.canseriald.panda_emulator import emulate_panda
from util import create_periodic_task
from typing import List

dbcPath = '../../odrivebody/odrive_comma_body.dbc'
db = cantools.database.load_file(dbcPath)

odrive0_node_id = 0
odrive1_node_id = 1

message_send_query: List[can.Message] = []


async def run_main():
  pm = messaging.PubMaster(["can", "pandaStates", "peripheralState"])
  sm = messaging.SubMaster(["sendcan", "can"])

  # pylint: disable=abstract-class-instantiated
  with can.interface.Bus(bustype='slcan', channel='/dev/ttyACM0', bitrate=250000) as bus:
    can.Notifier(bus, [handle_recieved_can_message], loop=asyncio.get_running_loop())
    # cleanup with notifier.stop()

    await asyncio.gather(
        create_periodic_task(lambda: emulate_panda(pm), 0.5),
        create_periodic_task(lambda: can_send_messages(sm, bus), frequency=500),
        create_periodic_task(lambda: publish_recieved_messages(pm), frequency=500),
        create_periodic_task(lambda: request_speed(bus), frequency=90),
        create_periodic_task(lambda: request_battery(bus), frequency=2)
        # create_periodic_task(lambda: log_original_can_messages(sm), frequency=100)
    )


def can_send_messages(sm, bus):
  sm.update(0)
  if sm.updated['sendcan']:
    for capnp in sm['sendcan']:
      bus.send(can_from_capnp(capnp))


def publish_recieved_messages(pm):
  pm.send("can", cans_to_capnp(message_send_query))
  message_send_query.clear()


def request_speed(bus):
  get_vel_Estimate_command = 9
  request_can_message(bus, odrive0_node_id, get_vel_Estimate_command)
  request_can_message(bus, odrive1_node_id, get_vel_Estimate_command)


def request_battery(bus):
  get_Vbus_Voltage_command = 23
  request_can_message(bus, odrive0_node_id, get_Vbus_Voltage_command)


def log_original_can_messages(sm):
  sm.update(0)
  if sm.updated['can']:
    for capnp in sm['can']:
      print(can_from_capnp(capnp))


def handle_recieved_can_message(message: can.Message) -> None:
  heartbeat_command_id = 1
  if not get_command_id(message.arbitration_id) == heartbeat_command_id:
    encoded_message = db.decode_message(
        message.arbitration_id, message.data)
    print(encoded_message)
  message_send_query.append(message)


def request_can_message(bus: can.interface.Bus, node_id: int, command_id: int) -> None:
  message = can.Message(arbitration_id=get_arbitration_id(node_id, command_id),
                        is_remote_frame=True, is_extended_id=False)
  bus.send(message)


def get_arbitration_id(node_id: int, command_id: int) -> int:
  return (node_id << 5) + command_id


def get_command_id(arbitration_id: int) -> int:
  return arbitration_id & 0b00011111


def can_from_capnp(capnp_message):
  return can.Message(arbitration_id=capnp_message.address,
                     data=bytes(capnp_message.dat), is_extended_id=False)


def cans_to_capnp(can_messages: List[can.Message]):
  msg = messaging.new_message('can', size=len(can_messages))

  for i, can_message in enumerate(can_messages):
    msg.can[i] = {"address": can_message.arbitration_id, "dat": bytes(can_message.data)}

  return msg


def main():
  asyncio.run(run_main())


if __name__ == "__main__":
  main()
