DEBUG = False

def msg(x):
  if DEBUG:
    print "S:",x.encode("hex")
  if len(x) <= 7:
    ret = chr(len(x)) + x
  else:
    assert False
  return ret.ljust(8, "\x00")

def isotp_send(panda, x, addr, bus=0):
  if len(x) <= 7:
    panda.can_send(addr, msg(x), bus)
  else:
    ss = chr(0x10 + (len(x)>>8)) + chr(len(x)&0xFF) + x[0:6]
    x = x[6:]
    idx = 1
    sends = []
    while len(x) > 0:
      sends.append(((chr(0x20 + (idx&0xF)) + x[0:7]).ljust(8, "\x00")))
      x = x[7:]
      idx += 1

    # actually send
    panda.can_send(addr, ss, bus)
    rr = recv(panda, 1, addr+8, bus)[0]
    panda.can_send_many([(addr, None, s, 0) for s in sends])

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
        pass
    kmsgs = nmsgs
  return map(str, ret)

def isotp_recv(panda, addr, bus=0):
  msg = recv(panda, 1, addr, bus)[0]

  if ord(msg[0])&0xf0 == 0x10:
    # first
    tlen = ((ord(msg[0]) & 0xf) << 8) | ord(msg[1])
    dat = msg[2:]

    # 0 block size?
    CONTINUE = "\x30" + "\x00"*7

    panda.can_send(addr-8, CONTINUE, bus)

    idx = 1
    for mm in recv(panda, (tlen-len(dat) + 7)/8, addr, bus):
      assert ord(mm[0]) == (0x20 | idx)
      dat += mm[1:]
      idx += 1
  elif ord(msg[0])&0xf0 == 0x00:
    # single
    tlen = ord(msg[0]) & 0xf
    dat = msg[1:]
  else:
    assert False

  dat = dat[0:tlen]

  if DEBUG:
    print "R:",dat.encode("hex")

  return dat

