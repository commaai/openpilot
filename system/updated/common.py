import os
import pathlib


def get_consistent_flag(path: str) -> bool:
  consistent_file = pathlib.Path(os.path.join(path, ".overlay_consistent"))
  return consistent_file.is_file()

def set_consistent_flag(path: str, consistent: bool) -> None:
  os.sync()
  consistent_file = pathlib.Path(os.path.join(path, ".overlay_consistent"))
  if consistent:
    consistent_file.touch()
  elif not consistent:
    consistent_file.unlink(missing_ok=True)
  os.sync()
