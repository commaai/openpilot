import zmq

from cereal import log
from common.realtime import sec_since_boot
from selfdrive.services import service_list

def new_message():
  dat = log.Event.new_message()
  dat.logMonoTime = int(sec_since_boot() * 1e9)
  dat.valid = True
  return dat

def pub_sock(port, addr="*"):
  context = zmq.Context.instance()
  sock = context.socket(zmq.PUB)
  sock.bind("tcp://%s:%d" % (addr, port))
  return sock

def sub_sock(port, poller=None, addr="127.0.0.1", conflate=False):
  context = zmq.Context.instance()
  sock = context.socket(zmq.SUB)
  if conflate:
    sock.setsockopt(zmq.CONFLATE, 1)
  sock.connect("tcp://%s:%d" % (addr, port))
  sock.setsockopt(zmq.SUBSCRIBE, b"")
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


class SubMaster():
  def __init__(self, services, addr="127.0.0.1"):
    self.poller = zmq.Poller()
    self.frame = -1
    self.updated = {s : False for s in services}
    self.rcv_time = {s : 0. for s in services}
    self.rcv_frame = {s : 0 for s in services}
    self.alive = {s : False for s in services}
    self.sock = {}
    self.freq = {}
    self.data = {}
    self.logMonoTime = {}
    self.valid = {}
    for s in services:
      # TODO: get address automatically from service_list
      self.sock[s] = sub_sock(service_list[s].port, poller=self.poller, addr=addr, conflate=True)
      self.freq[s] = service_list[s].frequency
      data = new_message()
      data.init(s)
      self.data[s] = getattr(data, s)
      self.logMonoTime[s] = data.logMonoTime
      self.valid[s] = data.valid

  def __getitem__(self, s):
    return self.data[s]

  def update(self, timeout=-1):
    # TODO: add optional input that specify the service to wait for
    self.frame += 1
    self.updated = dict.fromkeys(self.updated, False)
    cur_time = sec_since_boot()
    for sock, _ in self.poller.poll(timeout):
      msg = recv_one(sock)
      s = msg.which()
      self.updated[s] = True
      self.rcv_time[s] = cur_time
      self.rcv_frame[s] = self.frame
      self.data[s] = getattr(msg, s)
      self.logMonoTime[s] = msg.logMonoTime
      self.valid[s] = msg.valid

    for s in self.data:
      # arbitrary small number to avoid float comparison. If freq is 0, we can skip the check
      if self.freq[s] > 1e-5:
        # alive if delay is within 10x the expected frequency
        self.alive[s] = (cur_time - self.rcv_time[s]) < (10. / self.freq[s])
      else:
        self.alive[s] = True

  def all_alive(self, service_list=None):
    if service_list is None:  # check all
      service_list = self.alive.keys()
    return all(self.alive[s] for s in service_list)

  def all_valid(self, service_list=None):
    if service_list is None:  # check all
      service_list = self.valid.keys()
    return all(self.valid[s] for s in service_list)

  def all_alive_and_valid(self, service_list=None):
    if service_list is None:  # check all
      service_list = self.alive.keys()
    return self.all_alive(service_list=service_list) and self.all_valid(service_list=service_list)
