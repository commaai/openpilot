from typing import TYPE_CHECKING

# The UI also runs where the params extension isn't built (e.g. a PC without a build), so tolerate it being missing.
if TYPE_CHECKING:
  from openpilot.common.params import Params, UnknownKeyName
else:
  try:
    from openpilot.common.params import Params, UnknownKeyName
  except (ImportError, OSError):
    Params = None
    UnknownKeyName = Exception

__all__ = ["Params", "UnknownKeyName", "ui_params"]

_ui_params: "Params | None" = None


def ui_params() -> "Params":
  """One Params handle shared by the UI's param controls."""
  global _ui_params
  if _ui_params is None:
    assert Params is not None
    _ui_params = Params()
  return _ui_params
