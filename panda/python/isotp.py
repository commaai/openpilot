import binascii
import time

DEBUG = False

def msg(x):
  if DEBUG:
    print("S:", binascii.hexlify(x))
  if len(x) <= 7:
    ret = bytes([len(x)]) + x
  else:
    assert False
  return ret.ljust(8, b"\x00")

kmsgs = []
def recv(panda, cnt, addr, nbus):
  global kmsgs
  ret = []

  while len(ret) < cnt:
    kmsgs += panda.can_recv()
    nmsgs = []
    for ids, ts, dat, bus in kmsgs:
      if ids == addr and bus == nbus and len(ret) < cnt:
        ret.append(dat)
      else:
        # leave around
        nmsgs.append((ids, ts, dat, bus))
    kmsgs = nmsgs[-256:]
  return ret

def isotp_recv_subaddr(panda, addr, bus, sendaddr, subaddr):
  msg = recv(panda, 1, addr, bus)[0]

  # TODO: handle other subaddr also communicating
  assert msg[0] == subaddr

  if msg[1] & 0xf0 == 0x10:
    # first
    tlen = ((msg[1] & 0xf) << 8) | msg[2]
    dat = msg[3:]

    # 0 block size?
    CONTINUE = bytes([subaddr]) + b"\x30" + b"\x00" * 6
    panda.can_send(sendaddr, CONTINUE, bus)

    idx = 1
    for mm in recv(panda, (tlen - len(dat) + 5) // 6, addr, bus):
      assert mm[0] == subaddr
      assert mm[1] == (0x20 | (idx & 0xF))
      dat += mm[2:]
      idx += 1
  elif msg[1] & 0xf0 == 0x00:
    # single
    tlen = msg[1] & 0xf
    dat = msg[2:]
  else:
    print(binascii.hexlify(msg))
    assert False

  return dat[0:tlen]

# **** import below this line ****

def isotp_send(panda, x, addr, bus=0, recvaddr=None, subaddr=None, rate=None):
  if recvaddr is None:
    recvaddr = addr + 8

  if len(x) <= 7 and subaddr is None:
    panda.can_send(addr, msg(x), bus)
  elif len(x) <= 6 and subaddr is not None:
    panda.can_send(addr, bytes([subaddr]) + msg(x)[0:7], bus)
  else:
    if subaddr:
      ss = bytes([subaddr, 0x10 + (len(x) >> 8), len(x) & 0xFF]) + x[0:5]
      x = x[5:]
    else:
      ss = bytes([0x10 + (len(x) >> 8), len(x) & 0xFF]) + x[0:6]
      x = x[6:]
    idx = 1
    sends = []
    while len(x) > 0:
      if subaddr:
        sends.append(((bytes([subaddr, 0x20 + (idx & 0xF)]) + x[0:6]).ljust(8, b"\x00")))
        x = x[6:]
      else:
        sends.append(((bytes([0x20 + (idx & 0xF)]) + x[0:7]).ljust(8, b"\x00")))
        x = x[7:]
      idx += 1

    # actually send
    panda.can_send(addr, ss, bus)
    rr = recv(panda, 1, recvaddr, bus)[0]
    if rr.find(b"\x30\x01") != -1:
      for s in sends[:-1]:
        panda.can_send(addr, s, 0)
        rr = recv(panda, 1, recvaddr, bus)[0]
      panda.can_send(addr, sends[-1], 0)
    else:
      if rate is None:
        panda.can_send_many([(addr, None, s, bus) for s in sends])
      else:
        for dat in sends:
          panda.can_send(addr, dat, bus)
          time.sleep(rate)

def isotp_recv(panda, addr, bus=0, sendaddr=None, subaddr=None):
  if sendaddr is None:
    sendaddr = addr - 8

  if subaddr is not None:
    dat = isotp_recv_subaddr(panda, addr, bus, sendaddr, subaddr)
  else:
    msg = recv(panda, 1, addr, bus)[0]

    if msg[0] & 0xf0 == 0x10:
      # first
      tlen = ((msg[0] & 0xf) << 8) | msg[1]
      dat = msg[2:]

      # 0 block size?
      CONTINUE = b"\x30" + b"\x00" * 7

      panda.can_send(sendaddr, CONTINUE, bus)

      idx = 1
      for mm in recv(panda, (tlen - len(dat) + 6) // 7, addr, bus):
        assert mm[0] == (0x20 | (idx & 0xF))
        dat += mm[1:]
        idx += 1
    elif msg[0] & 0xf0 == 0x00:
      # single
      tlen = msg[0] & 0xf
      dat = msg[1:]
    else:
      assert False
    dat = dat[0:tlen]

  if DEBUG:
    print("R:", binascii.hexlify(dat))

  return dat
