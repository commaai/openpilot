from unittest.mock import Mock

import pytest

from openpilot.selfdrive.modeld.modeld import retry_model_load


def test_retry_model_load():
  params = Mock()
  params.get_bool.return_value = False
  error = RuntimeError("USB transfer failed")

  with pytest.raises(RuntimeError, match="USB transfer failed"):
    retry_model_load(params, error)
  params.put_bool.assert_called_once_with("ChestnutModelError", True)


@pytest.mark.parametrize("error", [None, RuntimeError("USB transfer failed")])
def test_model_load_fallback(error):
  params = Mock()
  params.get_bool.return_value = True

  retry_model_load(params, error)
  params.put_bool.assert_not_called()
