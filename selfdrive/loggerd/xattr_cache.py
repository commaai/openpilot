import os
from typing import Dict, Tuple

cached_attributes: Dict[Tuple, bytes] = {}
def getxattr(path: str, attr_name: str) -> bytes:
  if (path, attr_name) not in cached_attributes:
    response = os.getxattr(path, attr_name)
    cached_attributes[(path, attr_name)] = response
  return cached_attributes[(path, attr_name)]

def setxattr(path: str, attr_name: str, attr_value: bytes) -> None:
  cached_attributes.pop((path, attr_name), None)
  return os.setxattr(path, attr_name, attr_value)
