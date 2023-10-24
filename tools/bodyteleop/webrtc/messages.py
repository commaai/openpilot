import dataclasses


@dataclasses.dataclass
class PlayAlertMessage:
  alertType: str

@dataclasses.dataclass
class BatteryLevelMessage:
  batteryLevel: int

@dataclasses.dataclass
class JoystickMessage:
  x: int
  y: int

# one channel for everything
class Message:
  type: str
  message: object

  @staticmethod
  def from_dict(json_dict: dict):
    msg_type = json_dict["type"]
    msg_dict = json_dict["message"]
    if msg_type == "playAlert":
      return Message(msg_type, PlayAlertMessage(**msg_dict))
    elif msg_type == "batteryLevel":
      return Message(msg_type, BatteryLevelMessage(**msg_dict))
    elif msg_type == "joystick":
      return Message(msg_type, JoystickMessage(**msg_dict))
    else:
      raise Exception(f"Unknown message type {msg_type}")

  def to_dict(self) -> dict:
    msg_dict = dataclasses.asdict(self.message)
    return {"type": self.type, "message": msg_dict}

import json
from typing import List

import aiortc
import asyncio

from cereal import messaging


class CerealMessageProxy:
  def __init__(self, services: List[str]):
    self.services = services
    self.sm = messaging.SubMaster(self.services)
    self.channels = []
    self.is_running = False
    self.task = None

  def add_channel(self, channel: aiortc.RTCDataChannel):
    self.channels.append(channel)

  def start(self):
    assert not self.is_running
    self.task = asyncio.create_task(self.run())
    self.is_running = True

  def stop(self):
    assert self.is_running
    self.task.cancel()
    self.is_running = False

  async def run(self):
    while True:
      self.sm.update(0)
      for service, updated in self.sm.updated.items():
        if not updated:
          continue
        # dynamic struct reader as dictionary
        msg_dict, mono_time, valid = self.sm[service].to_dict(), self.sm.logMonoTime[service], self.sm.valid[service]
        outgoing_msg = {"type": service, "logMonoTime": mono_time, "valid": valid, "data": msg_dict}
        encoded_msg = json.dumps(outgoing_msg).encode()
        for channel in self.channels:
          await channel.send(encoded_msg)
      await asyncio.sleep(0.01)
