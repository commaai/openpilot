import os
import pathlib


USERDATA = os.getenv("USERDATA_DIR", "/data")
FINALIZED = os.path.join(USERDATA, "finalized")


def get_valid_flag(path: str) -> bool:
  valid_file = pathlib.Path(os.path.join(path, ".update_valid"))
  return valid_file.is_file()


def set_valid_flag(path: str, valid: bool) -> None:
  os.sync()
  valid_file = pathlib.Path(os.path.join(path, ".update_valid"))
  if valid:
    valid_file.touch()
  else:
    valid_file.unlink(missing_ok=True)
  os.sync()
