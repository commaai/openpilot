# capnpyx/wire.py
import struct

PTR_STRUCT = 0
PTR_LIST   = 1
PTR_FAR    = 2
PTR_CAP    = 3

def u16(buf, off): return int.from_bytes(buf[off:off+2], "little")
def u32(buf, off): return int.from_bytes(buf[off:off+4], "little")
def u64(buf, off): return int.from_bytes(buf[off:off+8], "little")
def i16(buf, off): return int.from_bytes(buf[off:off+2], "little", signed=True)
def i32(buf, off): return int.from_bytes(buf[off:off+4], "little", signed=True)
def i64(buf, off): return int.from_bytes(buf[off:off+8], "little", signed=True)

def f32(buf, off): return struct.unpack_from("<f", buf, off)[0]
def f64(buf, off): return struct.unpack_from("<d", buf, off)[0]

def read_ptr_word(buf, off): return u64(buf, off)
def ptr_kind(word): return word & 3
def ptr_offset_words(word):
    off = (word >> 2) & ((1<<30)-1)
    if off & (1<<29): off -= (1<<30)
    return off

def struct_data_size(word): return ((word >> 32) & 0xffff) * 8
def struct_ptr_count(word): return (word >> 48) & 0xffff
