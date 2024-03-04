from __future__ import absolute_import

import binascii
import weakref

from kafka.vendor import six


if six.PY3:
    MAX_INT = 2 ** 31
    TO_SIGNED = 2 ** 32

    def crc32(data):
        crc = binascii.crc32(data)
        # py2 and py3 behave a little differently
        # CRC is encoded as a signed int in kafka protocol
        # so we'll convert the py3 unsigned result to signed
        if crc >= MAX_INT:
            crc -= TO_SIGNED
        return crc
else:
    from binascii import crc32


class WeakMethod(object):
    """
    Callable that weakly references a method and the object it is bound to. It
    is based on https://stackoverflow.com/a/24287465.

    Arguments:

        object_dot_method: A bound instance method (i.e. 'object.method').
    """
    def __init__(self, object_dot_method):
        try:
            self.target = weakref.ref(object_dot_method.__self__)
        except AttributeError:
            self.target = weakref.ref(object_dot_method.im_self)
        self._target_id = id(self.target())
        try:
            self.method = weakref.ref(object_dot_method.__func__)
        except AttributeError:
            self.method = weakref.ref(object_dot_method.im_func)
        self._method_id = id(self.method())

    def __call__(self, *args, **kwargs):
        """
        Calls the method on target with args and kwargs.
        """
        return self.method()(self.target(), *args, **kwargs)

    def __hash__(self):
        return hash(self.target) ^ hash(self.method)

    def __eq__(self, other):
        if not isinstance(other, WeakMethod):
            return False
        return self._target_id == other._target_id and self._method_id == other._method_id


class Dict(dict):
    """Utility class to support passing weakrefs to dicts

    See: https://docs.python.org/2/library/weakref.html
    """
    pass
