# must be built with scons
from msgq.ipc_pyx import Context, Poller, SubSocket, PubSocket, SocketEventHandle, toggle_fake_events, \
                                set_fake_prefix, get_fake_prefix, delete_fake_prefix, wait_for_one_event
from msgq.ipc_pyx import MultiplePublishersError, IpcError
from msgq import fake_event_handle, pub_sock, sub_sock, drain_sock_raw, context

import os
import capnp
import time

from typing import Optional, List, Union, Dict, Deque
from collections import deque

from cereal import log
from cereal.services import SERVICE_LIST

NO_TRAVERSAL_LIMIT = 2**64-1


def log_from_bytes(dat: bytes) -> capnp.lib.capnp._DynamicStructReader:
  with log.Event.from_bytes(dat, traversal_limit_in_words=NO_TRAVERSAL_LIMIT) as msg:
    return msg


def new_message(service: Optional[str], size: Optional[int] = None, **kwargs) -> capnp.lib.capnp._DynamicStructBuilder:
  args = {
    'valid': False,
    'logMonoTime': int(time.monotonic() * 1e9),
    **kwargs
  }
  dat = log.Event.new_message(**args)
  if service is not None:
    if size is None:
      dat.init(service)
    else:
      dat.init(service, size)
  return dat


def drain_sock(sock: SubSocket, wait_for_one: bool = False) -> List[capnp.lib.capnp._DynamicStructReader]:
  """Receive all message currently available on the queue"""
  msgs = drain_sock_raw(sock, wait_for_one=wait_for_one)
  return [log_from_bytes(m) for m in msgs]


# TODO: print when we drop packets?
def recv_sock(sock: SubSocket, wait: bool = False) -> Optional[capnp.lib.capnp._DynamicStructReader]:
  """Same as drain sock, but only returns latest message. Consider using conflate instead."""
  dat = None

  while 1:
    if wait and dat is None:
      recv = sock.receive()
    else:
      recv = sock.receive(non_blocking=True)

    if recv is None:  # Timeout hit
      break

    dat = recv

  if dat is not None:
    dat = log_from_bytes(dat)

  return dat


def recv_one(sock: SubSocket) -> Optional[capnp.lib.capnp._DynamicStructReader]:
  dat = sock.receive()
  if dat is not None:
    dat = log_from_bytes(dat)
  return dat


def recv_one_or_none(sock: SubSocket) -> Optional[capnp.lib.capnp._DynamicStructReader]:
  dat = sock.receive(non_blocking=True)
  if dat is not None:
    dat = log_from_bytes(dat)
  return dat


def recv_one_retry(sock: SubSocket) -> capnp.lib.capnp._DynamicStructReader:
  """Keep receiving until we get a message"""
  while True:
    dat = sock.receive()
    if dat is not None:
      return log_from_bytes(dat)


class SubMaster:
  def __init__(self, services: List[str], poll: Optional[str] = None,
               ignore_alive: Optional[List[str]] = None, ignore_avg_freq: Optional[List[str]] = None,
               ignore_valid: Optional[List[str]] = None, addr: str = "127.0.0.1", frequency: Optional[float] = None):
    self.frame = -1
    self.seen = {s: False for s in services}
    self.updated = {s: False for s in services}
    self.recv_time = {s: 0. for s in services}
    self.recv_frame = {s: 0 for s in services}
    self.alive = {s: False for s in services}
    self.freq_ok = {s: False for s in services}
    self.recv_dts: Dict[str, Deque[float]] = {}
    self.sock = {}
    self.data = {}
    self.valid = {}
    self.logMonoTime = {}

    self.max_freq = {}
    self.min_freq = {}

    self.poller = Poller()
    polled_services = set([poll, ] if poll is not None else services)
    self.non_polled_services = set(services) - polled_services

    self.ignore_average_freq = [] if ignore_avg_freq is None else ignore_avg_freq
    self.ignore_alive = [] if ignore_alive is None else ignore_alive
    self.ignore_valid = [] if ignore_valid is None else ignore_valid

    self.simulation = bool(int(os.getenv("SIMULATION", "0")))

    # if freq and poll aren't specified, assume the max to be conservative
    assert frequency is None or poll is None, "Do not specify 'frequency' - frequency of the polled service will be used."
    self.update_freq = frequency or max([SERVICE_LIST[s].frequency for s in polled_services])

    for s in services:
      p = self.poller if s not in self.non_polled_services else None
      self.sock[s] = sub_sock(s, poller=p, addr=addr, conflate=True)

      try:
        data = new_message(s)
      except capnp.lib.capnp.KjException:
        data = new_message(s, 0) # lists

      self.data[s] = getattr(data.as_reader(), s)
      self.logMonoTime[s] = 0
      self.valid[s] = True  # FIXME: this should default to False

      freq = max(min([SERVICE_LIST[s].frequency, self.update_freq]), 1.)
      if s == poll:
        max_freq = freq
        min_freq = freq
      else:
        max_freq = min(freq, self.update_freq)
        if SERVICE_LIST[s].frequency >= 2*self.update_freq:
          min_freq = self.update_freq
        elif self.update_freq >= 2*SERVICE_LIST[s].frequency:
          min_freq = freq
        else:
          min_freq = min(freq, freq / 2.)
      self.max_freq[s] = max_freq*1.2
      self.min_freq[s] = min_freq*0.8
      self.recv_dts[s] = deque(maxlen=int(10*freq))

  def __getitem__(self, s: str) -> capnp.lib.capnp._DynamicStructReader:
    return self.data[s]

  def _check_avg_freq(self, s: str) -> bool:
    return SERVICE_LIST[s].frequency > 0.99 and (s not in self.ignore_average_freq) and (s not in self.ignore_alive)

  def update(self, timeout: int = 100) -> None:
    msgs = []
    for sock in self.poller.poll(timeout):
      msgs.append(recv_one_or_none(sock))

    # non-blocking receive for non-polled sockets
    for s in self.non_polled_services:
      msgs.append(recv_one_or_none(self.sock[s]))
    self.update_msgs(time.monotonic(), msgs)

  def update_msgs(self, cur_time: float, msgs: List[capnp.lib.capnp._DynamicStructReader]) -> None:
    self.frame += 1
    self.updated = dict.fromkeys(self.updated, False)
    for msg in msgs:
      if msg is None:
        continue

      s = msg.which()
      self.seen[s] = True
      self.updated[s] = True

      if self.recv_time[s] > 1e-5:
        self.recv_dts[s].append(cur_time - self.recv_time[s])
      self.recv_time[s] = cur_time
      self.recv_frame[s] = self.frame
      self.data[s] = getattr(msg, s)
      self.logMonoTime[s] = msg.logMonoTime
      self.valid[s] = msg.valid

    for s in self.data:
      if SERVICE_LIST[s].frequency > 1e-5 and not self.simulation:
        # alive if delay is within 10x the expected frequency
        self.alive[s] = (cur_time - self.recv_time[s]) < (10. / SERVICE_LIST[s].frequency)

        # check average frequency; slow to fall, quick to recover
        dts = self.recv_dts[s]
        assert dts.maxlen is not None
        recent_dts = list(dts)[-int(dts.maxlen / 10):]
        try:
          avg_freq = 1 / (sum(dts) / len(dts))
          avg_freq_recent = 1 / (sum(recent_dts) / len(recent_dts))
        except ZeroDivisionError:
          avg_freq = 0
          avg_freq_recent = 0

        avg_freq_ok = self.min_freq[s] <= avg_freq <= self.max_freq[s]
        recent_freq_ok = self.min_freq[s] <= avg_freq_recent <= self.max_freq[s]
        self.freq_ok[s] = avg_freq_ok or recent_freq_ok
      else:
        self.freq_ok[s] = True
        if self.simulation:
          self.alive[s] = self.seen[s] # alive is defined as seen when simulation flag set
        else:
          self.alive[s] = True

  def all_alive(self, service_list: Optional[List[str]] = None) -> bool:
    if service_list is None:
      service_list = list(self.sock.keys())
    return all(self.alive[s] for s in service_list if s not in self.ignore_alive)

  def all_freq_ok(self, service_list: Optional[List[str]] = None) -> bool:
    if service_list is None:
      service_list = list(self.sock.keys())
    return all(self.freq_ok[s] for s in service_list if self._check_avg_freq(s))

  def all_valid(self, service_list: Optional[List[str]] = None) -> bool:
    if service_list is None:
      service_list = list(self.sock.keys())
    return all(self.valid[s] for s in service_list if s not in self.ignore_valid)

  def all_checks(self, service_list: Optional[List[str]] = None) -> bool:
    return self.all_alive(service_list) and self.all_freq_ok(service_list) and self.all_valid(service_list)


class PubMaster:
  def __init__(self, services: List[str]):
    self.sock = {}
    for s in services:
      self.sock[s] = pub_sock(s)

  def send(self, s: str, dat: Union[bytes, capnp.lib.capnp._DynamicStructBuilder]) -> None:
    if not isinstance(dat, bytes):
      dat = dat.to_bytes()
    self.sock[s].send(dat)

  def wait_for_readers_to_update(self, s: str, timeout: int, dt: float = 0.05) -> bool:
    for _ in range(int(timeout*(1./dt))):
      if self.sock[s].all_readers_updated():
        return True
      time.sleep(dt)
    return False

  def all_readers_updated(self, s: str) -> bool:
    return self.sock[s].all_readers_updated()  # type: ignore
