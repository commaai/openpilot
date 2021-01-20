#!/usr/bin/env python3

import os
import sys
import struct
import time
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda, PandaWifiStreaming  # noqa: E402

# test throughput between USB and wifi

if __name__ == "__main__":
  print(Panda.list())
  p_out = Panda("108018800f51363038363036")
  print(p_out.get_serial())
  p_in = Panda("WIFI")
  print(p_in.get_serial())

  p_in = PandaWifiStreaming()  # type: ignore

  p_out.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  set_out, set_in = set(), set()

  # drain
  p_out.can_recv()
  p_in.can_recv()

  BATCH_SIZE = 16
  for a in tqdm(list(range(0, 10000, BATCH_SIZE))):
    for b in range(0, BATCH_SIZE):
      msg = b"\xaa" * 4 + struct.pack("I", a + b)
      if a % 1 == 0:
        p_out.can_send(0xaa, msg, 0)

    dat_out, dat_in = p_out.can_recv(), p_in.can_recv()
    if len(dat_in) != 0:
      print(len(dat_in))

    num_out = [struct.unpack("I", i[4:])[0] for _, _, i, _ in dat_out]
    num_in = [struct.unpack("I", i[4:])[0] for _, _, i, _ in dat_in]

    set_in.update(num_in)
    set_out.update(num_out)

  # swag
  print("waiting for packets")
  time.sleep(2.0)
  dat_in = p_in.can_recv()
  print(len(dat_in))
  num_in = [struct.unpack("I", i[4:])[0] for _, _, i, _ in dat_in]
  set_in.update(num_in)

  if len(set_out - set_in):
    print("MISSING %d" % len(set_out - set_in))
    if len(set_out - set_in) < 256:
      print(list(map(hex, sorted(list(set_out - set_in)))))
