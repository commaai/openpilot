# cereal_schema.py (auto-gen later)
from cereal.capnpy.reader import Reader
from cereal.capnpy.builder import Builder

class Event(Reader):
    __slots__ = ()
    def eventType(self): return self._data(0, 16, signed=False)
    def logMonoTime(self): return self._data(8, 64, signed=False)
    # example ptr
    def carState(self): return self._ptr(0, CarState)

class CarState(Reader):
    __slots__ = ()
    def vEgo(self): return self._data(0, 32, float_=True)
    def aEgo(self): return self._data(4, 32, float_=True)
