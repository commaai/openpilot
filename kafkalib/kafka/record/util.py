import binascii

from kafka.record._crc32c import crc as crc32c_py
try:
    from crc32c import crc32c as crc32c_c
except ImportError:
    crc32c_c = None


def encode_varint(value, write):
    """ Encode an integer to a varint presentation. See
    https://developers.google.com/protocol-buffers/docs/encoding?csw=1#varints
    on how those can be produced.

        Arguments:
            value (int): Value to encode
            write (function): Called per byte that needs to be writen

        Returns:
            int: Number of bytes written
    """
    value = (value << 1) ^ (value >> 63)

    if value <= 0x7f:  # 1 byte
        write(value)
        return 1
    if value <= 0x3fff:  # 2 bytes
        write(0x80 | (value & 0x7f))
        write(value >> 7)
        return 2
    if value <= 0x1fffff:  # 3 bytes
        write(0x80 | (value & 0x7f))
        write(0x80 | ((value >> 7) & 0x7f))
        write(value >> 14)
        return 3
    if value <= 0xfffffff:  # 4 bytes
        write(0x80 | (value & 0x7f))
        write(0x80 | ((value >> 7) & 0x7f))
        write(0x80 | ((value >> 14) & 0x7f))
        write(value >> 21)
        return 4
    if value <= 0x7ffffffff:  # 5 bytes
        write(0x80 | (value & 0x7f))
        write(0x80 | ((value >> 7) & 0x7f))
        write(0x80 | ((value >> 14) & 0x7f))
        write(0x80 | ((value >> 21) & 0x7f))
        write(value >> 28)
        return 5
    else:
        # Return to general algorithm
        bits = value & 0x7f
        value >>= 7
        i = 0
        while value:
            write(0x80 | bits)
            bits = value & 0x7f
            value >>= 7
            i += 1
    write(bits)
    return i


def size_of_varint(value):
    """ Number of bytes needed to encode an integer in variable-length format.
    """
    value = (value << 1) ^ (value >> 63)
    if value <= 0x7f:
        return 1
    if value <= 0x3fff:
        return 2
    if value <= 0x1fffff:
        return 3
    if value <= 0xfffffff:
        return 4
    if value <= 0x7ffffffff:
        return 5
    if value <= 0x3ffffffffff:
        return 6
    if value <= 0x1ffffffffffff:
        return 7
    if value <= 0xffffffffffffff:
        return 8
    if value <= 0x7fffffffffffffff:
        return 9
    return 10


def decode_varint(buffer, pos=0):
    """ Decode an integer from a varint presentation. See
    https://developers.google.com/protocol-buffers/docs/encoding?csw=1#varints
    on how those can be produced.

        Arguments:
            buffer (bytearray): buffer to read from.
            pos (int): optional position to read from

        Returns:
            (int, int): Decoded int value and next read position
    """
    result = buffer[pos]
    if not (result & 0x81):
        return (result >> 1), pos + 1
    if not (result & 0x80):
        return (result >> 1) ^ (~0), pos + 1

    result &= 0x7f
    pos += 1
    shift = 7
    while 1:
        b = buffer[pos]
        result |= ((b & 0x7f) << shift)
        pos += 1
        if not (b & 0x80):
            return ((result >> 1) ^ -(result & 1), pos)
        shift += 7
        if shift >= 64:
            raise ValueError("Out of int64 range")


_crc32c = crc32c_py
if crc32c_c is not None:
    _crc32c = crc32c_c


def calc_crc32c(memview, _crc32c=_crc32c):
    """ Calculate CRC-32C (Castagnoli) checksum over a memoryview of data
    """
    return _crc32c(memview)


def calc_crc32(memview):
    """ Calculate simple CRC-32 checksum over a memoryview of data
    """
    crc = binascii.crc32(memview) & 0xffffffff
    return crc
