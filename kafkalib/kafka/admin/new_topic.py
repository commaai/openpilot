from __future__ import absolute_import

from kafka.errors import IllegalArgumentError


class NewTopic(object):
    """ A class for new topic creation
    Arguments:
        name (string): name of the topic
        num_partitions (int): number of partitions
            or -1 if replica_assignment has been specified
        replication_factor (int): replication factor or -1 if
            replica assignment is specified
        replica_assignment (dict of int: [int]): A mapping containing
            partition id and replicas to assign to it.
        topic_configs (dict of str: str): A mapping of config key
            and value for the topic.
    """

    def __init__(
            self,
            name,
            num_partitions,
            replication_factor,
            replica_assignments=None,
            topic_configs=None,
    ):
        if not (num_partitions == -1 or replication_factor == -1) ^ (replica_assignments is None):
            raise IllegalArgumentError('either num_partitions/replication_factor or replica_assignment must be specified')
        self.name = name
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor
        self.replica_assignments = replica_assignments or {}
        self.topic_configs = topic_configs or {}
