"""Simple test file to demonstrate pytest caching optimization."""

def test_simple_function():
    """A simple test that should pass."""
    assert 1 + 1 == 2

def test_another_function():
    """Another simple test."""
    assert "hello".upper() == "HELLO"

def test_math_operations():
    """Test basic math operations."""
    assert 10 * 2 == 20
    assert 15 / 3 == 5

def test_list_operations():
    """Test list operations."""
    my_list = [1, 2, 3]
    my_list.append(4)
    assert len(my_list) == 4
    assert my_list[-1] == 4
