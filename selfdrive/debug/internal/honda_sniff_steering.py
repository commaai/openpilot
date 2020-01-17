#!/usr/bin/env python3
import os
import zmq

import cereal.messaging as messaging
from cereal.services import service_list

from panda.lib.panda import Panda
from hexdump import hexdump
import time

def raw_panda():
  p = Panda()
  print(p)

  p.set_uart_baud(2, 9600)
  p.set_uart_baud(3, 9600)

  p.set_uart_parity(2, 1)
  p.set_uart_parity(3, 1)

  p.set_uart_callback(2, 1)
  p.set_uart_callback(3, 1)

  idx = 0
  while 1:
    """
    dat = p.serial_read(2)
    if len(dat) > 0:
      print "2:",
      hexdump(dat)

    dat = p.serial_read(3)
    if len(dat) > 0:
      print "3:",
      hexdump(dat)

    print "read done, waiting"
    time.sleep(0.01)
    """

    if idx%2 == 1:
      dat = "\x20\x80\xc0\xa0"
    else:
      dat = "\x00\x80\xc0\xc0"
    p.can_send(0, dat, 8)

    for r in p.can_recv():
      if r[-1] in [8, 9]:
        print(r[-1], r[2].encode("hex"))

    time.sleep(0.01)
    idx += 1

if __name__ == "__main__":
  #raw_panda()
  #exit(0)

  logcan = messaging.sub_sock('can')

  t1 = []
  t2 = []
  t3 = []

  while len(t1) < 1000 or os.uname()[-1] == "aarch64":
    rr = messaging.recv_sock(logcan, wait=True)
    for c in rr.can:
      if c.src in [9] and len(c.dat) == 5:
        aa = map(lambda x: ord(x)&0x7f, c.dat)

        # checksum
        assert (-(aa[0]+aa[1]+aa[2]+aa[3]))&0x7f == aa[4]

        #print map(bin, aa[0:4])

        aa[0] &= ~0x20
        aa[1] &= ~0x20

        st = (aa[0] << 5) + aa[1]
        if st >= 256:
          st = -(512-st)

        mt = ((aa[2] >> 3) << 7) + aa[3]
        if mt >= 512:
          mt = -(1024-mt)

        print(st, mt)
        t1.append(st)
        t2.append(mt)
        #print map(bin, aa), "apply", st

      if c.src in [8] and len(c.dat) == 4:
        aa = map(lambda x: ord(x)&0x7f, c.dat)

        # checksum
        assert (-(aa[0]+aa[1]+aa[2]))&0x7f == aa[3]

        aa[0] &= ~0x20
        aa[1] &= ~0x20

        st = (aa[0] << 5) + aa[1]
        if st >= 256:
          st = -(512-st)
        print(aa, "apply", st)

        t3.append(st)

  import matplotlib.pyplot as plt
  plt.plot(t1)
  plt.plot(t2)
  plt.plot(t3)
  plt.show()

