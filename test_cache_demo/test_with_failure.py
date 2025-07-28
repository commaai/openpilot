"""Test file with some failing tests to demonstrate cache optimization."""


def test_passing_1():
    """This test should pass."""
    assert True


def test_failing():
    """This test was previously failing but is now fixed."""
    assert True, "This test now passes"


def test_passing_2():
    """Another passing test."""
    assert 2 + 2 == 4
