from ..manager import JSONRPCResponseManager

import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestJSONRPCResponseManager(unittest.TestCase):
    @unittest.skipIf(sys.version_info < (3, 5), "Test Py3.5+ functionality")
    def test_typeerror_with_annotations(self):
        """If a function has Python3 annotations and is called with improper
        arguments, make sure the framework doesn't fail with inspect.getargspec
        """
        from .py35_utils import distance

        dispatcher = {
            "distance": distance,
        }

        req = '{"jsonrpc": "2.0", "method": "distance", "params": [], "id": 1}'
        result = JSONRPCResponseManager.handle(req, dispatcher)

        # Make sure this returns JSONRPCInvalidParams rather than raising
        # UnboundLocalError
        self.assertEqual(result.error['code'], -32602)
