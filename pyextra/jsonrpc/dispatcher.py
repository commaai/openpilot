""" Dispatcher is used to add methods (functions) to the server.

For usage examples see :meth:`Dispatcher.add_method`

"""
import functools
import collections


class Dispatcher(collections.MutableMapping):

    """ Dictionary like object which maps method_name to method."""

    def __init__(self, prototype=None):
        """ Build method dispatcher.

        Parameters
        ----------
        prototype : object or dict, optional
            Initial method mapping.

        Examples
        --------

        Init object with method dictionary.

        >>> Dispatcher({"sum": lambda a, b: a + b})
        None

        """
        self.method_map = dict()

        if prototype is not None:
            self.build_method_map(prototype)

    def __getitem__(self, key):
        return self.method_map[key]

    def __setitem__(self, key, value):
        self.method_map[key] = value

    def __delitem__(self, key):
        del self.method_map[key]

    def __len__(self):
        return len(self.method_map)

    def __iter__(self):
        return iter(self.method_map)

    def __repr__(self):
        return repr(self.method_map)

    def add_class(self, cls):
        prefix = cls.__name__.lower() + '.'
        self.build_method_map(cls(), prefix)

    def add_object(self, obj):
        prefix = obj.__class__.__name__.lower() + '.'
        self.build_method_map(obj, prefix)

    def add_dict(self, dict, prefix=''):
        if prefix:
            prefix += '.'
        self.build_method_map(dict, prefix)

    def add_method(self, f=None, name=None):
        """ Add a method to the dispatcher.

        Parameters
        ----------
        f : callable
            Callable to be added.
        name : str, optional
            Name to register (the default is function **f** name)

        Notes
        -----
        When used as a decorator keeps callable object unmodified.

        Examples
        --------

        Use as method

        >>> d = Dispatcher()
        >>> d.add_method(lambda a, b: a + b, name="sum")
        <function __main__.<lambda>>

        Or use as decorator

        >>> d = Dispatcher()
        >>> @d.add_method
            def mymethod(*args, **kwargs):
                print(args, kwargs)

        Or use as a decorator with a different function name
        >>> d = Dispatcher()
        >>> @d.add_method(name="my.method")
            def mymethod(*args, **kwargs):
                print(args, kwargs)

        """
        if name and not f:
            return functools.partial(self.add_method, name=name)

        self.method_map[name or f.__name__] = f
        return f

    def build_method_map(self, prototype, prefix=''):
        """ Add prototype methods to the dispatcher.

        Parameters
        ----------
        prototype : object or dict
            Initial method mapping.
            If given prototype is a dictionary then all callable objects will
            be added to dispatcher.
            If given prototype is an object then all public methods will
            be used.
        prefix: string, optional
            Prefix of methods

        """
        if not isinstance(prototype, dict):
            prototype = dict((method, getattr(prototype, method))
                             for method in dir(prototype)
                             if not method.startswith('_'))

        for attr, method in prototype.items():
            if callable(method):
                self[prefix + attr] = method
