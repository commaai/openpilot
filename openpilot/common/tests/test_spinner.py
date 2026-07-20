import os
from unittest.mock import patch

from openpilot.common.spinner import Spinner


def test_spinner_disabled_in_ci():
  with patch.dict(os.environ, {"CI": "1"}), patch("subprocess.Popen") as popen:
    spinner = Spinner()
    spinner.update("building")
    spinner.close()

  popen.assert_not_called()
