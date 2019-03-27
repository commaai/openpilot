from ..dispatcher import Dispatcher
import sys
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class Math:

    def sum(self, a, b):
        return a + b

    def diff(self, a, b):
        return a - b


class TestDispatcher(unittest.TestCase):

    """ Test Dispatcher functionality."""

    def test_getter(self):
        d = Dispatcher()

        with self.assertRaises(KeyError):
            d["method"]

        d["add"] = lambda *args: sum(args)
        self.assertEqual(d["add"](1, 1), 2)

    def test_in(self):
        d = Dispatcher()
        d["method"] = lambda: ""
        self.assertIn("method", d)

    def test_add_method(self):
        d = Dispatcher()

        @d.add_method
        def add(x, y):
            return x + y

        self.assertIn("add", d)
        self.assertEqual(d["add"](1, 1), 2)

    def test_add_method_with_name(self):
        d = Dispatcher()

        @d.add_method(name="this.add")
        def add(x, y):
            return x + y

        self.assertNotIn("add", d)
        self.assertIn("this.add", d)
        self.assertEqual(d["this.add"](1, 1), 2)

    def test_add_class(self):
        d = Dispatcher()
        d.add_class(Math)

        self.assertIn("math.sum", d)
        self.assertIn("math.diff", d)
        self.assertEqual(d["math.sum"](3, 8), 11)
        self.assertEqual(d["math.diff"](6, 9), -3)

    def test_add_object(self):
        d = Dispatcher()
        d.add_object(Math())

        self.assertIn("math.sum", d)
        self.assertIn("math.diff", d)
        self.assertEqual(d["math.sum"](5, 2), 7)
        self.assertEqual(d["math.diff"](15, 9), 6)

    def test_add_dict(self):
        d = Dispatcher()
        d.add_dict({"sum": lambda *args: sum(args)}, "util")

        self.assertIn("util.sum", d)
        self.assertEqual(d["util.sum"](13, -2), 11)

    def test_add_method_keep_function_definitions(self):

        d = Dispatcher()

        @d.add_method
        def one(x):
            return x

        self.assertIsNotNone(one)

    def test_del_method(self):
        d = Dispatcher()
        d["method"] = lambda: ""
        self.assertIn("method", d)

        del d["method"]
        self.assertNotIn("method", d)

    def test_to_dict(self):
        d = Dispatcher()

        def func():
            return ""

        d["method"] = func
        self.assertEqual(dict(d), {"method": func})

    def test_init_from_object_instance(self):

        class Dummy():

            def one(self):
                pass

            def two(self):
                pass

        dummy = Dummy()

        d = Dispatcher(dummy)

        self.assertIn("one", d)
        self.assertIn("two", d)
        self.assertNotIn("__class__", d)

    def test_init_from_dictionary(self):

        dummy = {
            'one': lambda x: x,
            'two': lambda x: x,
        }

        d = Dispatcher(dummy)

        self.assertIn("one", d)
        self.assertIn("two", d)

    def test_dispatcher_representation(self):

        d = Dispatcher()
        self.assertEqual('{}', repr(d))
