# Mock capnp module for testing purposes
class MockCapnp:
    pass

# Mock for the capnp module to avoid import errors during testing
import sys
sys.modules['capnp'] = MockCapnp()
