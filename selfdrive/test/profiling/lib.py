from collections import defaultdict
from cereal.services import SERVICE_LIST
import cereal.messaging as messaging
import capnp


class ReplayDone(Exception):
  pass


class SubSocket:
  def __init__(self, msgs, trigger):
    self.i = 0
    self.trigger = trigger
    self.msgs = [m.as_builder().to_bytes() for m in msgs if m.which() == trigger]
    self.max_i = len(self.msgs) - 1

  def receive(self, non_blocking=False):
    if non_blocking:
      return None

    if self.i == self.max_i:
      raise ReplayDone

    while True:
      msg = self.msgs[self.i]
      self.i += 1
      return msg


class PubSocket:
  def send(self, data):
    pass


class SubMaster(messaging.SubMaster):
  def __init__(self, msgs, trigger, services, check_averag_freq=False):
    self.frame = 0
    self.data = {}
    self.ignore_alive = []

    self.alive = {s: True for s in services}
    self.updated = {s: False for s in services}
    self.rcv_time = {s: 0. for s in services}
    self.rcv_frame = {s: 0 for s in services}
    self.valid = {s: True for s in services}
    self.freq_ok = {s: True for s in services}
    self.freq_tracker = {s: messaging.FrequencyTracker(SERVICE_LIST[s].frequency, SERVICE_LIST[s].frequency, False) for s in services}
    self.logMonoTime = {}
    self.sock = {}
    self.freq = {}
    self.check_average_freq = check_averag_freq
    self.non_polled_services = []
    self.ignore_average_freq = []

    # TODO: specify multiple triggers for service like plannerd that poll on more than one service
    cur_msgs = []
    self.msgs = []
    msgs = [m for m in msgs if m.which() in services]

    for msg in msgs:
      cur_msgs.append(msg)
      if msg.which() == trigger:
        self.msgs.append(cur_msgs)
        cur_msgs = []

    self.msgs = list(reversed(self.msgs))

    for s in services:
      self.freq[s] = SERVICE_LIST[s].frequency
      try:
        data = messaging.new_message(s)
      except capnp.lib.capnp.KjException:
        # lists
        data = messaging.new_message(s, 0)

      self.data[s] = getattr(data, s)
      self.logMonoTime[s] = 0
      self.sock[s] = SubSocket(msgs, s)

  def update(self, timeout=None):
    if not len(self.msgs):
      raise ReplayDone

    cur_msgs = self.msgs.pop()
    self.update_msgs(cur_msgs[0].logMonoTime, self.msgs.pop())


class PubMaster(messaging.PubMaster):
  def __init__(self):
    self.sock = defaultdict(PubSocket)
