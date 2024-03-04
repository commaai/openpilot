from __future__ import absolute_import


class NewPartitions(object):
    """A class for new partition creation on existing topics.  Note that the length of new_assignments, if specified,
    must be the difference between the new total number of partitions and the existing number of partitions.
    Arguments:
        total_count (int): the total number of partitions that should exist on the topic
        new_assignments ([[int]]): an array of arrays of replica assignments for new partitions.
            If not set, broker assigns replicas per an internal algorithm.
    """

    def __init__(
        self,
        total_count,
        new_assignments=None
    ):
        self.total_count = total_count
        self.new_assignments = new_assignments
