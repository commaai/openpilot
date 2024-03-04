from __future__ import absolute_import

import gzip
import io
import platform
import struct

from kafka.vendor import six
from kafka.vendor.six.moves import range

_XERIAL_V1_HEADER = (-126, b'S', b'N', b'A', b'P', b'P', b'Y', 0, 1, 1)
_XERIAL_V1_FORMAT = 'bccccccBii'
ZSTD_MAX_OUTPUT_SIZE = 1024 * 1024

try:
    import snappy
except ImportError:
    snappy = None

try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import lz4.frame as lz4

    def _lz4_compress(payload, **kwargs):
        # Kafka does not support LZ4 dependent blocks
        try:
            # For lz4>=0.12.0
            kwargs.pop('block_linked', None)
            return lz4.compress(payload, block_linked=False, **kwargs)
        except TypeError:
            # For earlier versions of lz4
            kwargs.pop('block_mode', None)
            return lz4.compress(payload, block_mode=1, **kwargs)

except ImportError:
    lz4 = None

try:
    import lz4f
except ImportError:
    lz4f = None

try:
    import lz4framed
except ImportError:
    lz4framed = None

try:
    import xxhash
except ImportError:
    xxhash = None

PYPY = bool(platform.python_implementation() == 'PyPy')

def has_gzip():
    return True


def has_snappy():
    return snappy is not None


def has_zstd():
    return zstd is not None


def has_lz4():
    if lz4 is not None:
        return True
    if lz4f is not None:
        return True
    if lz4framed is not None:
        return True
    return False


def gzip_encode(payload, compresslevel=None):
    if not compresslevel:
        compresslevel = 9

    buf = io.BytesIO()

    # Gzip context manager introduced in python 2.7
    # so old-fashioned way until we decide to not support 2.6
    gzipper = gzip.GzipFile(fileobj=buf, mode="w", compresslevel=compresslevel)
    try:
        gzipper.write(payload)
    finally:
        gzipper.close()

    return buf.getvalue()


def gzip_decode(payload):
    buf = io.BytesIO(payload)

    # Gzip context manager introduced in python 2.7
    # so old-fashioned way until we decide to not support 2.6
    gzipper = gzip.GzipFile(fileobj=buf, mode='r')
    try:
        return gzipper.read()
    finally:
        gzipper.close()


def snappy_encode(payload, xerial_compatible=True, xerial_blocksize=32*1024):
    """Encodes the given data with snappy compression.

    If xerial_compatible is set then the stream is encoded in a fashion
    compatible with the xerial snappy library.

    The block size (xerial_blocksize) controls how frequent the blocking occurs
    32k is the default in the xerial library.

    The format winds up being:


        +-------------+------------+--------------+------------+--------------+
        |   Header    | Block1 len | Block1 data  | Blockn len | Blockn data  |
        +-------------+------------+--------------+------------+--------------+
        |  16 bytes   |  BE int32  | snappy bytes |  BE int32  | snappy bytes |
        +-------------+------------+--------------+------------+--------------+


    It is important to note that the blocksize is the amount of uncompressed
    data presented to snappy at each block, whereas the blocklen is the number
    of bytes that will be present in the stream; so the length will always be
    <= blocksize.

    """

    if not has_snappy():
        raise NotImplementedError("Snappy codec is not available")

    if not xerial_compatible:
        return snappy.compress(payload)

    out = io.BytesIO()
    for fmt, dat in zip(_XERIAL_V1_FORMAT, _XERIAL_V1_HEADER):
        out.write(struct.pack('!' + fmt, dat))

    # Chunk through buffers to avoid creating intermediate slice copies
    if PYPY:
        # on pypy, snappy.compress() on a sliced buffer consumes the entire
        # buffer... likely a python-snappy bug, so just use a slice copy
        chunker = lambda payload, i, size: payload[i:size+i]

    elif six.PY2:
        # Sliced buffer avoids additional copies
        # pylint: disable-msg=undefined-variable
        chunker = lambda payload, i, size: buffer(payload, i, size)
    else:
        # snappy.compress does not like raw memoryviews, so we have to convert
        # tobytes, which is a copy... oh well. it's the thought that counts.
        # pylint: disable-msg=undefined-variable
        chunker = lambda payload, i, size: memoryview(payload)[i:size+i].tobytes()

    for chunk in (chunker(payload, i, xerial_blocksize)
                  for i in range(0, len(payload), xerial_blocksize)):

        block = snappy.compress(chunk)
        block_size = len(block)
        out.write(struct.pack('!i', block_size))
        out.write(block)

    return out.getvalue()


def _detect_xerial_stream(payload):
    """Detects if the data given might have been encoded with the blocking mode
        of the xerial snappy library.

        This mode writes a magic header of the format:
            +--------+--------------+------------+---------+--------+
            | Marker | Magic String | Null / Pad | Version | Compat |
            +--------+--------------+------------+---------+--------+
            |  byte  |   c-string   |    byte    |  int32  | int32  |
            +--------+--------------+------------+---------+--------+
            |  -126  |   'SNAPPY'   |     \0     |         |        |
            +--------+--------------+------------+---------+--------+

        The pad appears to be to ensure that SNAPPY is a valid cstring
        The version is the version of this format as written by xerial,
        in the wild this is currently 1 as such we only support v1.

        Compat is there to claim the miniumum supported version that
        can read a xerial block stream, presently in the wild this is
        1.
    """

    if len(payload) > 16:
        header = struct.unpack('!' + _XERIAL_V1_FORMAT, bytes(payload)[:16])
        return header == _XERIAL_V1_HEADER
    return False


def snappy_decode(payload):
    if not has_snappy():
        raise NotImplementedError("Snappy codec is not available")

    if _detect_xerial_stream(payload):
        # TODO ? Should become a fileobj ?
        out = io.BytesIO()
        byt = payload[16:]
        length = len(byt)
        cursor = 0

        while cursor < length:
            block_size = struct.unpack_from('!i', byt[cursor:])[0]
            # Skip the block size
            cursor += 4
            end = cursor + block_size
            out.write(snappy.decompress(byt[cursor:end]))
            cursor = end

        out.seek(0)
        return out.read()
    else:
        return snappy.decompress(payload)


if lz4:
    lz4_encode = _lz4_compress # pylint: disable-msg=no-member
elif lz4f:
    lz4_encode = lz4f.compressFrame # pylint: disable-msg=no-member
elif lz4framed:
    lz4_encode = lz4framed.compress # pylint: disable-msg=no-member
else:
    lz4_encode = None


def lz4f_decode(payload):
    """Decode payload using interoperable LZ4 framing. Requires Kafka >= 0.10"""
    # pylint: disable-msg=no-member
    ctx = lz4f.createDecompContext()
    data = lz4f.decompressFrame(payload, ctx)
    lz4f.freeDecompContext(ctx)

    # lz4f python module does not expose how much of the payload was
    # actually read if the decompression was only partial.
    if data['next'] != 0:
        raise RuntimeError('lz4f unable to decompress full payload')
    return data['decomp']


if lz4:
    lz4_decode = lz4.decompress # pylint: disable-msg=no-member
elif lz4f:
    lz4_decode = lz4f_decode
elif lz4framed:
    lz4_decode = lz4framed.decompress # pylint: disable-msg=no-member
else:
    lz4_decode = None


def lz4_encode_old_kafka(payload):
    """Encode payload for 0.8/0.9 brokers -- requires an incorrect header checksum."""
    assert xxhash is not None
    data = lz4_encode(payload)
    header_size = 7
    flg = data[4]
    if not isinstance(flg, int):
        flg = ord(flg)

    content_size_bit = ((flg >> 3) & 1)
    if content_size_bit:
        # Old kafka does not accept the content-size field
        # so we need to discard it and reset the header flag
        flg -= 8
        data = bytearray(data)
        data[4] = flg
        data = bytes(data)
        payload = data[header_size+8:]
    else:
        payload = data[header_size:]

    # This is the incorrect hc
    hc = xxhash.xxh32(data[0:header_size-1]).digest()[-2:-1]  # pylint: disable-msg=no-member

    return b''.join([
        data[0:header_size-1],
        hc,
        payload
    ])


def lz4_decode_old_kafka(payload):
    assert xxhash is not None
    # Kafka's LZ4 code has a bug in its header checksum implementation
    header_size = 7
    if isinstance(payload[4], int):
        flg = payload[4]
    else:
        flg = ord(payload[4])
    content_size_bit = ((flg >> 3) & 1)
    if content_size_bit:
        header_size += 8

    # This should be the correct hc
    hc = xxhash.xxh32(payload[4:header_size-1]).digest()[-2:-1]  # pylint: disable-msg=no-member

    munged_payload = b''.join([
        payload[0:header_size-1],
        hc,
        payload[header_size:]
    ])
    return lz4_decode(munged_payload)


def zstd_encode(payload):
    if not zstd:
        raise NotImplementedError("Zstd codec is not available")
    return zstd.ZstdCompressor().compress(payload)


def zstd_decode(payload):
    if not zstd:
        raise NotImplementedError("Zstd codec is not available")
    try:
        return zstd.ZstdDecompressor().decompress(payload)
    except zstd.ZstdError:
        return zstd.ZstdDecompressor().decompress(payload, max_output_size=ZSTD_MAX_OUTPUT_SIZE)
