import pytest

from openpilot.common.prefix import OpenpilotPrefix


@pytest.fixture(scope="function", autouse=True)
def global_setup_and_teardown():
  with OpenpilotPrefix():
    yield
