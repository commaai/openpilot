# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import sys
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc cimport errno
from libc.string cimport strerror
from cython.operator import dereference


from .messaging cimport Context as cppContext
from .messaging cimport SubSocket as cppSubSocket
from .messaging cimport PubSocket as cppPubSocket
from .messaging cimport Poller as cppPoller
from .messaging cimport Message as cppMessage
from .messaging cimport Event as cppEvent, SocketEventHandle as cppSocketEventHandle


class MessagingError(Exception):
  def __init__(self, endpoint=None):
    suffix = "with {endpoint}" if endpoint else ""
    message = f"Messaging failure {suffix}: {strerror(errno.errno)}"
    super().__init__(message)


class MultiplePublishersError(MessagingError):
  pass


def toggle_fake_events(bool enabled):
  cppSocketEventHandle.toggle_fake_events(enabled)


def set_fake_prefix(string prefix):
  cppSocketEventHandle.set_fake_prefix(prefix)


def get_fake_prefix():
  return cppSocketEventHandle.fake_prefix()


def delete_fake_prefix():
  cppSocketEventHandle.set_fake_prefix(b"")


def wait_for_one_event(list events, int timeout=-1):
  cdef vector[cppEvent] items
  for event in events:
    items.push_back(dereference(<cppEvent*><size_t>event.ptr))
  return cppEvent.wait_for_one(items, timeout)


cdef class Event:
  cdef cppEvent event;

  def __cinit__(self):
    pass

  cdef setEvent(self, cppEvent event):
    self.event = event

  def set(self):
    self.event.set()

  def clear(self):
    return self.event.clear()

  def wait(self, int timeout=-1):
    self.event.wait(timeout)

  def peek(self):
    return self.event.peek()

  @property
  def fd(self):
    return self.event.fd()

  @property
  def ptr(self):
    return <size_t><void*>&self.event


cdef class SocketEventHandle:
  cdef cppSocketEventHandle * handle;

  def __cinit__(self, string endpoint, string identifier, bool override):
    self.handle = new cppSocketEventHandle(endpoint, identifier, override)

  def __dealloc__(self):
    del self.handle

  @property
  def enabled(self):
    return self.handle.is_enabled()

  @enabled.setter
  def enabled(self, bool value):
    self.handle.set_enabled(value)

  @property
  def recv_called_event(self):
    e = Event()
    e.setEvent(self.handle.recv_called())

    return e

  @property
  def recv_ready_event(self):
    e = Event()
    e.setEvent(self.handle.recv_ready())

    return e


cdef class Context:
  cdef cppContext * context

  def __cinit__(self):
    self.context = cppContext.create()

  def term(self):
    del self.context
    self.context = NULL

  def __dealloc__(self):
    pass
    # Deleting the context will hang if sockets are still active
    # TODO: Figure out a way to make sure the context is closed last
    # del self.context


cdef class Poller:
  cdef cppPoller * poller
  cdef list sub_sockets

  def __cinit__(self):
    self.sub_sockets = []
    self.poller = cppPoller.create()

  def __dealloc__(self):
    del self.poller

  def registerSocket(self, SubSocket socket):
    self.sub_sockets.append(socket)
    self.poller.registerSocket(socket.socket)

  def poll(self, timeout):
    sockets = []
    cdef int t = timeout

    with nogil:
      result = self.poller.poll(t)

    for s in result:
      socket = SubSocket()
      socket.setPtr(s)
      sockets.append(socket)

    return sockets


cdef class SubSocket:
  cdef cppSubSocket * socket
  cdef bool is_owner

  def __cinit__(self):
    self.socket = cppSubSocket.create()
    self.is_owner = True

    if self.socket == NULL:
      raise MessagingError

  def __dealloc__(self):
    if self.is_owner:
      del self.socket

  cdef setPtr(self, cppSubSocket * ptr):
    if self.is_owner:
      del self.socket

    self.is_owner = False
    self.socket = ptr

  def connect(self, Context context, string endpoint, string address=b"127.0.0.1", bool conflate=False):
    r = self.socket.connect(context.context, endpoint, address, conflate)

    if r != 0:
      if errno.errno == errno.EADDRINUSE:
        raise MultiplePublishersError(endpoint)
      else:
        raise MessagingError(endpoint)

  def setTimeout(self, int timeout):
    self.socket.setTimeout(timeout)

  def receive(self, bool non_blocking=False):
    msg = self.socket.receive(non_blocking)

    if msg == NULL:
      # If a blocking read returns no message check errno if SIGINT was caught in the C++ code
      if errno.errno == errno.EINTR:
        print("SIGINT received, exiting")
        sys.exit(1)

      return None
    else:
      sz = msg.getSize()
      m = msg.getData()[:sz]
      del msg

      return m


cdef class PubSocket:
  cdef cppPubSocket * socket

  def __cinit__(self):
    self.socket = cppPubSocket.create()
    if self.socket == NULL:
      raise MessagingError

  def __dealloc__(self):
    del self.socket

  def connect(self, Context context, string endpoint):
    r = self.socket.connect(context.context, endpoint)

    if r != 0:
      if errno.errno == errno.EADDRINUSE:
        raise MultiplePublishersError(endpoint)
      else:
        raise MessagingError(endpoint)

  def send(self, bytes data):
    length = len(data)
    r = self.socket.send(<char*>data, length)

    if r != length:
      if errno.errno == errno.EADDRINUSE:
        raise MultiplePublishersError
      else:
        raise MessagingError

  def all_readers_updated(self):
    return self.socket.all_readers_updated()
