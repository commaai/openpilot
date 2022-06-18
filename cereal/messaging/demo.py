import time

from messaging_pyx import Context, Poller, SubSocket, PubSocket  # pylint: disable=no-name-in-module, import-error

MSGS = 1e5

if __name__ == "__main__":
  c = Context()
  sub_sock = SubSocket()
  pub_sock = PubSocket()

  sub_sock.connect(c, "controlsState")
  pub_sock.connect(c, "controlsState")

  poller = Poller()
  poller.registerSocket(sub_sock)

  t = time.time()
  for i in range(int(MSGS)):
    bts = i.to_bytes(4, 'little')
    pub_sock.send(bts)

    for s in poller.poll(100):
      dat = s.receive()
      ii = int.from_bytes(dat, 'little')
      assert(i == ii)

  dt = time.time() - t
  print("%.1f msg/s" % (MSGS / dt))
