import zmq

from cereal import log
from common import realtime

def new_message():
  dat = log.Event.new_message()
  dat.logMonoTime = int(realtime.sec_since_boot() * 1e9)
  return dat

def pub_sock(context, port, addr="*"):
  sock = context.socket(zmq.PUB)
  sock.bind("tcp://%s:%d" % (addr, port))
  return sock

def sub_sock(context, port, poller=None, addr="127.0.0.1", conflate=False):
  sock = context.socket(zmq.SUB)
  if conflate:
    sock.setsockopt(zmq.CONFLATE, 1)
  sock.connect("tcp://%s:%d" % (addr, port))
  sock.setsockopt(zmq.SUBSCRIBE, "")
  if poller is not None:
    poller.register(sock, zmq.POLLIN)
  return sock

def drain_sock(sock, wait_for_one=False):
  ret = []
  while 1:
    try:
      if wait_for_one and len(ret) == 0:
        dat = sock.recv()
      else:
        dat = sock.recv(zmq.NOBLOCK)
      dat = log.Event.from_bytes(dat)
      ret.append(dat)
    except zmq.error.Again:
      break
  return ret


# TODO: print when we drop packets?
def recv_sock(sock, wait=False):
  dat = None
  while 1:
    try:
      if wait and dat is None:
        dat = sock.recv()
      else:
        dat = sock.recv(zmq.NOBLOCK)
    except zmq.error.Again:
      break
  if dat is not None:
    dat = log.Event.from_bytes(dat)
  return dat

def recv_one(sock):
  return log.Event.from_bytes(sock.recv())

def recv_one_or_none(sock):
  try:
    return log.Event.from_bytes(sock.recv(zmq.NOBLOCK))
  except zmq.error.Again:
    return None
