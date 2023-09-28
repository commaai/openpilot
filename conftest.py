import pytest

from openpilot.common.prefix import OpenpilotPrefix


@pytest.fixture(scope="function", autouse=True)
def global_setup_and_teardown():
  # setup a clean environment for each test
  with OpenpilotPrefix():
    yield

# there is currently no way to mark cpp tests, so we have to mark them here
ADDITIONAL_EXPLICIT = ["test_translations", "test_sound"]

def pytest_collection_modifyitems(config, items):
  skipper = pytest.mark.skip(reason="Skipping explicit test since it was not run directly.")
  for item in items:
    test_filename = item.fspath.basename
    is_explicit_test = "explicit" in item.keywords or test_filename in ADDITIONAL_EXPLICIT
    was_run_explicitly = test_filename in config.option.file_or_dir
    if is_explicit_test and not was_run_explicitly:
      item.add_marker(skipper)