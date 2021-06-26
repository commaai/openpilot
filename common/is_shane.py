# mypy: ignore-errors
import importlib

try:
  Params = importlib.import_module('common.params').Params
  is_shane = Params().get("DongleId").decode('utf8') in ['14431dbeedbf3558', 'e010b634f3d65cdb']  # returns if fork owner is current user
except Exception:  # in case params isn't built
  is_shane = False
