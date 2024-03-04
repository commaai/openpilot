from __future__ import absolute_import

from .base import Partitioner


class RoundRobinPartitioner(Partitioner):
    def __init__(self, partitions=None):
        self.partitions_iterable = CachedPartitionCycler(partitions)
        if partitions:
            self._set_partitions(partitions)
        else:
            self.partitions = None

    def __call__(self, key, all_partitions=None, available_partitions=None):
        if available_partitions:
            cur_partitions = available_partitions
        else:
            cur_partitions = all_partitions
        if not self.partitions:
            self._set_partitions(cur_partitions)
        elif cur_partitions != self.partitions_iterable.partitions and cur_partitions is not None:
            self._set_partitions(cur_partitions)
        return next(self.partitions_iterable)

    def _set_partitions(self, available_partitions):
        self.partitions = available_partitions
        self.partitions_iterable.set_partitions(available_partitions)

    def partition(self, key, all_partitions=None, available_partitions=None):
        return self.__call__(key, all_partitions, available_partitions)


class CachedPartitionCycler(object):
    def __init__(self, partitions=None):
        self.partitions = partitions
        if partitions:
            assert type(partitions) is list
        self.cur_pos = None

    def __next__(self):
        return self.next()

    @staticmethod
    def _index_available(cur_pos, partitions):
        return cur_pos < len(partitions)

    def set_partitions(self, partitions):
        if self.cur_pos:
            if not self._index_available(self.cur_pos, partitions):
                self.cur_pos = 0
                self.partitions = partitions
                return None

            self.partitions = partitions
            next_item = self.partitions[self.cur_pos]
            if next_item in partitions:
                self.cur_pos = partitions.index(next_item)
            else:
                self.cur_pos = 0
            return None
        self.partitions = partitions

    def next(self):
        assert self.partitions is not None
        if self.cur_pos is None or not self._index_available(self.cur_pos, self.partitions):
            self.cur_pos = 1
            return self.partitions[0]
        cur_item = self.partitions[self.cur_pos]
        self.cur_pos += 1
        return cur_item
