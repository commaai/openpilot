from __future__ import absolute_import

import abc

# This statement is compatible with both Python 2.7 & 3+
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class AbstractTokenProvider(ABC):
    """
    A Token Provider must be used for the SASL OAuthBearer protocol.

    The implementation should ensure token reuse so that multiple
    calls at connect time do not create multiple tokens. The implementation
    should also periodically refresh the token in order to guarantee
    that each call returns an unexpired token. A timeout error should
    be returned after a short period of inactivity so that the
    broker can log debugging info and retry.

    Token Providers MUST implement the token() method
    """

    def __init__(self, **config):
        pass

    @abc.abstractmethod
    def token(self):
        """
        Returns a (str) ID/Access Token to be sent to the Kafka
        client.
        """
        pass

    def extensions(self):
        """
        This is an OPTIONAL method that may be implemented.

        Returns a map of key-value pairs that can
        be sent with the SASL/OAUTHBEARER initial client request. If
        not implemented, the values are ignored. This feature is only available
        in Kafka >= 2.1.0.
        """
        return {}
