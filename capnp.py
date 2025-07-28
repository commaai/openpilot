# Mock capnp module for testing purposes
class MockSchema:
    """Mock Cap'n Proto schema"""
    def __init__(self, name="MockSchema"):
        self.name = name
    
    def __getattr__(self, name):
        # Return a mock class for any schema access
        return type(name, (), {})

class MockCapnp:
    def remove_import_hook(self):
        """Mock implementation of remove_import_hook"""
        pass
    
    def load(self, file_path):
        """Mock implementation of capnp.load"""
        return MockSchema(file_path)

# Mock for the capnp module to avoid import errors during testing
import sys
sys.modules['capnp'] = MockCapnp()
