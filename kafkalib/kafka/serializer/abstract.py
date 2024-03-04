from __future__ import absolute_import

import abc


class Serializer(object):
    __meta__ = abc.ABCMeta

    def __init__(self, **config):
        pass

    @abc.abstractmethod
    def serialize(self, topic, value):
        pass

    def close(self):
        pass


class Deserializer(object):
    __meta__ = abc.ABCMeta

    def __init__(self, **config):
        pass

    @abc.abstractmethod
    def deserialize(self, topic, bytes_):
        pass

    def close(self):
        pass
