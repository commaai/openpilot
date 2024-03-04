from __future__ import absolute_import

from kafka.vendor import six

from .base import Partitioner


class Murmur2Partitioner(Partitioner):
    """
    Implements a partitioner which selects the target partition based on
    the hash of the key. Attempts to apply the same hashing
    function as mainline java client.
    """
    def __call__(self, key, partitions=None, available=None):
        if available:
            return self.partition(key, available)
        return self.partition(key, partitions)

    def partition(self, key, partitions=None):
        if not partitions:
            partitions = self.partitions

        # https://github.com/apache/kafka/blob/0.8.2/clients/src/main/java/org/apache/kafka/clients/producer/internals/Partitioner.java#L69
        idx = (murmur2(key) & 0x7fffffff) % len(partitions)

        return partitions[idx]


class LegacyPartitioner(object):
    """DEPRECATED -- See Issue 374

    Implements a partitioner which selects the target partition based on
    the hash of the key
    """
    def __init__(self, partitions):
        self.partitions = partitions

    def partition(self, key, partitions=None):
        if not partitions:
            partitions = self.partitions
        size = len(partitions)
        idx = hash(key) % size

        return partitions[idx]


# Default will change to Murmur2 in 0.10 release
HashedPartitioner = LegacyPartitioner


# https://github.com/apache/kafka/blob/0.8.2/clients/src/main/java/org/apache/kafka/common/utils/Utils.java#L244
def murmur2(data):
    """Pure-python Murmur2 implementation.

    Based on java client, see org.apache.kafka.common.utils.Utils.murmur2

    Args:
        data (bytes): opaque bytes

    Returns: MurmurHash2 of data
    """
    # Python2 bytes is really a str, causing the bitwise operations below to fail
    # so convert to bytearray.
    if six.PY2:
        data = bytearray(bytes(data))

    length = len(data)
    seed = 0x9747b28c
    # 'm' and 'r' are mixing constants generated offline.
    # They're not really 'magic', they just happen to work well.
    m = 0x5bd1e995
    r = 24

    # Initialize the hash to a random value
    h = seed ^ length
    length4 = length // 4

    for i in range(length4):
        i4 = i * 4
        k = ((data[i4 + 0] & 0xff) +
            ((data[i4 + 1] & 0xff) << 8) +
            ((data[i4 + 2] & 0xff) << 16) +
            ((data[i4 + 3] & 0xff) << 24))
        k &= 0xffffffff
        k *= m
        k &= 0xffffffff
        k ^= (k % 0x100000000) >> r # k ^= k >>> r
        k &= 0xffffffff
        k *= m
        k &= 0xffffffff

        h *= m
        h &= 0xffffffff
        h ^= k
        h &= 0xffffffff

    # Handle the last few bytes of the input array
    extra_bytes = length % 4
    if extra_bytes >= 3:
        h ^= (data[(length & ~3) + 2] & 0xff) << 16
        h &= 0xffffffff
    if extra_bytes >= 2:
        h ^= (data[(length & ~3) + 1] & 0xff) << 8
        h &= 0xffffffff
    if extra_bytes >= 1:
        h ^= (data[length & ~3] & 0xff)
        h &= 0xffffffff
        h *= m
        h &= 0xffffffff

    h ^= (h % 0x100000000) >> 13 # h >>> 13;
    h &= 0xffffffff
    h *= m
    h &= 0xffffffff
    h ^= (h % 0x100000000) >> 15 # h >>> 15;
    h &= 0xffffffff

    return h
