# capnpyx/builder.py
import struct
from cereal.capnpy import wire as W

class Builder:
    __slots__ = ("_buf", "_base", "_data_size", "_ptr_count")

    def __init__(self, buf, base, data_size, ptr_count):
        self._buf = buf
        self._base = base
        self._data_size = data_size
        self._ptr_count = ptr_count

    def _set_data(self, off, bits, val, signed=False, float_=False):
        if float_:
            fmt = "<f" if bits == 32 else "<d"
            struct.pack_into(fmt, self._buf, self._base+off, val)
        else:
            self._buf[self._base+off:self._base+off+bits//8] = val.to_bytes(bits//8, "little", signed=signed)
