# capnpyx/reader.py
from cereal.capnpy import wire as W

class Reader:
    __slots__ = ("_buf", "_base", "_data_size", "_ptr_count", "_cache")

    def __init__(self, buf, base, data_size, ptr_count):
        self._buf = buf
        self._base = base
        self._data_size = data_size
        self._ptr_count = ptr_count
        self._cache = {}

    def _data(self, off, bits, signed=False, float_=False):
        k = (off, bits, signed, float_)
        if k in self._cache:
            return self._cache[k]
        if float_:
            val = W.f32(self._buf, self._base+off) if bits==32 else W.f64(self._buf, self._base+off)
        else:
            val = {
                (8,False): lambda: self._buf[self._base+off],
                (8,True):  lambda: int.from_bytes(self._buf[self._base+off:self._base+off+1], "little", signed=True),
                (16,False): lambda: W.u16(self._buf, self._base+off),
                (16,True):  lambda: W.i16(self._buf, self._base+off),
                (32,False): lambda: W.u32(self._buf, self._base+off),
                (32,True):  lambda: W.i32(self._buf, self._base+off),
                (64,False): lambda: W.u64(self._buf, self._base+off),
                (64,True):  lambda: W.i64(self._buf, self._base+off),
            }[(bits, signed)]()
        self._cache[k] = val
        return val

    def _ptr(self, idx, cls):
        k = ("ptr", idx)
        if k in self._cache:
            return self._cache[k]
        ptr_off = self._base + self._data_size + idx*8
        word = W.read_ptr_word(self._buf, ptr_off)
        if W.ptr_kind(word) == W.PTR_STRUCT:
            base = ptr_off + 8 + W.ptr_offset_words(word)*8
            reader = cls(self._buf, base, W.struct_data_size(word), W.struct_ptr_count(word))
        else:
            reader = None
        self._cache[k] = reader
        return reader
