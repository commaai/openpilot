#!/usr/bin/env python3
import os
import usb1
import time
import struct
import itertools
import threading
from typing import Any

from panda import Panda

JUNGLE = "JUNGLE" in os.environ
if JUNGLE:
  from panda import PandaJungle

# Generate unique messages
NUM_MESSAGES_PER_BUS = 10000
messages = [bytes(struct.pack("Q", i)) for i in range(NUM_MESSAGES_PER_BUS)]
tx_messages = list(itertools.chain.from_iterable([[0xaa, None, msg, 0], [0xaa, None, msg, 1], [0xaa, None, msg, 2]] for msg in messages))

def flood_tx(panda):
  print('Sending!')
  transferred = 0
  while True:
    try:
      print(f"Sending block {transferred}-{len(tx_messages)}: ", end="")
      panda.can_send_many(tx_messages[transferred:], timeout=10)
      print("OK")
      break
    except usb1.USBErrorTimeout as e:
      transferred += (e.transferred // 16)
      print("timeout, transferred: ", transferred)

  print(f"Done sending {3*NUM_MESSAGES_PER_BUS} messages!")

if __name__ == "__main__":
  serials = Panda.list()
  receiver: Panda | PandaJungle
  if JUNGLE:
    sender = Panda()
    receiver = PandaJungle()
  else:
    if len(serials) != 2:
      raise Exception("Connect two pandas to perform this test!")
    sender = Panda(serials[0])
    receiver = Panda(serials[1])
    receiver.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  sender.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  # Start transmisson
  threading.Thread(target=flood_tx, args=(sender,)).start()

  # Receive as much as we can, and stop when there hasn't been anything for a second
  rx: list[Any] = []
  old_len = 0
  last_change = time.monotonic()
  while time.monotonic() - last_change < 1:
    if old_len < len(rx):
      last_change = time.monotonic()
    old_len = len(rx)

    rx.extend(receiver.can_recv())
  print(f"Received {len(rx)} messages")

  # Check if we received everything
  for bus in range(3):
    received_msgs = {bytes(m[2]) for m in filter(lambda m, b=bus: m[3] == b, rx)} # type: ignore
    dropped_msgs = set(messages).difference(received_msgs)
    print(f"Bus {bus} dropped msgs: {len(list(dropped_msgs))} / {len(messages)}")
