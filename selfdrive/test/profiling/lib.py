from collections import defaultdict
import cereal.messaging as messaging
import capnp


class ReplayDone(Exception):
  pass


class SubSocket():
  def __init__(self, msgs, trigger):
    self.i = 0
    self.trigger = trigger
    self.msgs = [m for m in msgs if m.which() == trigger]
    self.max_i = len(self.msgs) - 1

  def receive(self, non_blocking=False):
    if non_blocking:
      return None

    if self.i == self.max_i:
      raise ReplayDone

    while True:
      msg = self.msgs[self.i]
      msg = msg.as_builder()

      self.i += 1

      return msg.to_bytes()


class PubSocket():
  def send(self, data):
    pass


class SubMaster():
  def __init__(self, msgs, trigger, services):
    self.max_i = len(msgs) - 1
    self.i = 0
    self.frame = 0
    self.trigger = trigger
    self.msgs = msgs
    self.data = {}

    self.alive = {s: True for s in services}
    self.updated = {s: False for s in services}
    self.rcv_time = {s: 0. for s in services}
    self.rcv_frame = {s: 0 for s in services}
    self.valid = {s: True for s in services}
    self.logMonoTime = {}
    self.sock = {}

    for s in services:
      try:
        data = messaging.new_message(s)
      except capnp.lib.capnp.KjException:
        # lists
        data = messaging.new_message(s, 0)

      self.data[s] = getattr(data, s)
      self.logMonoTime[s] = 0
      self.sock[s] = SubSocket(msgs, s)

  def update(self, timeout=None):
    if self.i == self.max_i:
      raise ReplayDone

    self.updated = dict.fromkeys(self.updated, False)
    self.frame += 1

    while True:
      msg = self.msgs[self.i]
      w = msg.which()

      self.updated[w] = True
      self.rcv_time[w] = msg.logMonoTime / 1e9
      self.rcv_frame[w] = self.frame
      self.data[w] = getattr(msg, w)
      self.logMonoTime[w] = msg.logMonoTime

      self.i += 1

      if w == self.trigger:
        break

  def all_alive(self):
    return True

  def all_valid(self):
    return True

  def all_alive_and_valid(self):
    return True

  def __getitem__(self, s):
    return self.data[s]


class PubMaster():
  def __init__(self):
    self.sock = defaultdict(PubSocket)

  def send(self, s, dat):
    pass
