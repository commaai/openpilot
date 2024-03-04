from __future__ import absolute_import

import abc
import logging

log = logging.getLogger(__name__)


class AbstractPartitionAssignor(object):
    """
    Abstract assignor implementation which does some common grunt work (in particular collecting
    partition counts which are always needed in assignors).
    """

    @abc.abstractproperty
    def name(self):
        """.name should be a string identifying the assignor"""
        pass

    @abc.abstractmethod
    def assign(self, cluster, members):
        """Perform group assignment given cluster metadata and member subscriptions

        Arguments:
            cluster (ClusterMetadata): metadata for use in assignment
            members (dict of {member_id: MemberMetadata}): decoded metadata for
                each member in the group.

        Returns:
            dict: {member_id: MemberAssignment}
        """
        pass

    @abc.abstractmethod
    def metadata(self, topics):
        """Generate ProtocolMetadata to be submitted via JoinGroupRequest.

        Arguments:
            topics (set): a member's subscribed topics

        Returns:
            MemberMetadata struct
        """
        pass

    @abc.abstractmethod
    def on_assignment(self, assignment):
        """Callback that runs on each assignment.

        This method can be used to update internal state, if any, of the
        partition assignor.

        Arguments:
            assignment (MemberAssignment): the member's assignment
        """
        pass
