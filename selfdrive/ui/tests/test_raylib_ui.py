import time
import pytest
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.system.ui.lib.multilang import multilang
from openpilot.common.params import Params


@pytest.fixture(autouse=True)
def set_language_param(language_code):
  Params().put("LanguageSetting", language_code)


@pytest.mark.parametrize("language_code", list(multilang.codes.keys()), ids=list(multilang.codes.keys()))
@with_processes(["ui"])
def test_raylib_ui(language_code):
  """Test initialization of the UI widgets is successful."""
  time.sleep(1)
