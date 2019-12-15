# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from libcpp.string cimport string
from libcpp cimport bool


from messaging cimport Context as cppContext
from messaging cimport SubSocket as cppSubSocket
from messaging cimport PubSocket as cppPubSocket
from messaging cimport Poller as cppPoller
from messaging cimport Message as cppMessage


cdef class Context:
  cdef cppContext * context
  def  __cinit__(self):
    self.context = cppContext.create()

  def __dealloc__(self):
    pass
    # Deleting the context will hang if sockets are still active
    # TODO: Figure out a way to make sure the context is closed last
    # del self.context


cdef class Poller:
  cdef cppPoller * poller
  cdef list sub_sockets

  def  __cinit__(self):
    self.sub_sockets = []
    self.poller = cppPoller.create()

  def __dealloc__(self):
    del self.poller

  def registerSocket(self, SubSocket socket):
    self.sub_sockets.append(socket)
    self.poller.registerSocket(socket.socket)

  def poll(self, timeout):
    sockets = []

    result = self.poller.poll(timeout)
    for s in result:
        socket = SubSocket()
        socket.setPtr(s)
        sockets.append(socket)

    return sockets

cdef class SubSocket:
  cdef cppSubSocket * socket
  cdef bool is_owner

  def  __cinit__(self):
    self.socket = cppSubSocket.create()
    self.is_owner = True

  def __dealloc__(self):
    if self.is_owner:
      del self.socket

  cdef setPtr(self, cppSubSocket * ptr):
    if self.is_owner:
      del self.socket

    self.is_owner = False
    self.socket = ptr

  def connect(self, Context context, string endpoint, string address=b"127.0.0.1", bool conflate=False):
    self.socket.connect(context.context, endpoint, address, conflate)

  def setTimeout(self, int timeout):
    self.socket.setTimeout(timeout)


  def receive(self, bool non_blocking=False):
    msg = self.socket.receive(non_blocking)

    if msg == NULL:
      return None
    else:
      sz = msg.getSize()
      m = msg.getData()[:sz]
      del msg

      return m


cdef class PubSocket:
  cdef cppPubSocket * socket
  def  __cinit__(self):
    self.socket = cppPubSocket.create()

  def __dealloc__(self):
    del self.socket

  def connect(self, Context context, string endpoint):
    self.socket.connect(context.context, endpoint)

  def send(self, string data):
    return self.socket.send(<char*>data.c_str(), len(data))
