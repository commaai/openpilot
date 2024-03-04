from __future__ import absolute_import

try:
    import copyreg  # pylint: disable=import-error
except ImportError:
    import copy_reg as copyreg  # pylint: disable=import-error

import types


def _pickle_method(method):
    try:
        func_name = method.__func__.__name__
        obj = method.__self__
        cls = method.__self__.__class__
    except AttributeError:
        func_name = method.im_func.__name__
        obj = method.im_self
        cls = method.im_class

    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
        return func.__get__(obj, cls)

# https://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
