from typing import Dict, Tuple

from common.xattr import getxattr as getattr1
from common.xattr import setxattr as setattr1

cached_attributes: Dict[Tuple, bytes] = {}
def getxattr(path: str, attr_name: bytes) -> bytes:
  if (path, attr_name) not in cached_attributes:
    response = getattr1(path, attr_name)
    cached_attributes[(path, attr_name)] = response
  return cached_attributes[(path, attr_name)]

def setxattr(path: str, attr_name: str, attr_value: bytes) -> None:
  cached_attributes.pop((path, attr_name), None)
  return setattr1(path, attr_name, attr_value)
