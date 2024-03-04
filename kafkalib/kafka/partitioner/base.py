from __future__ import absolute_import


class Partitioner(object):
    """
    Base class for a partitioner
    """
    def __init__(self, partitions=None):
        """
        Initialize the partitioner

        Arguments:
            partitions: A list of available partitions (during startup) OPTIONAL.
        """
        self.partitions = partitions

    def __call__(self, key, all_partitions=None, available_partitions=None):
        """
        Takes a string key, num_partitions and available_partitions as argument and returns
        a partition to be used for the message

        Arguments:
            key: the key to use for partitioning.
            all_partitions: a list of the topic's partitions.
            available_partitions: a list of the broker's currently avaliable partitions(optional).
        """
        raise NotImplementedError('partition function has to be implemented')
