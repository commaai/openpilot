import os
import pytest

from openpilot.common.prefix import OpenpilotPrefix


# there is currently no way to mark cpp tests, so we have to mark them here
ADDITIONAL_EXPLICIT = ["test_translations", "test_sound"]

def pytest_collection_modifyitems(config, items):
  skipper = pytest.mark.skip(reason="Skipping explicit test since it was not run directly.")
  for item in items:
    test_filename = item.fspath.basename
    is_explicit_test = "explicit" in item.keywords or test_filename in ADDITIONAL_EXPLICIT
    was_run_explicitly = any(test_filename in file for file in config.option.file_or_dir)
    if is_explicit_test and not was_run_explicitly:
      item.add_marker(skipper)


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture():
  starting_env = dict(os.environ)

  # setup a clean environment for each test
  with OpenpilotPrefix():
    yield

  os.environ.clear()
  os.environ.update(starting_env)


# If you use setUpClass, the environment variables won't be cleared properly,
# so we need to hook both the function and class pytest fixtures
@pytest.fixture(scope="class", autouse=True)
def openpilot_class_fixture():
  starting_env = dict(os.environ)

  yield

  os.environ.clear()
  os.environ.update(starting_env)
