from common.xattr import getxattr as getattr1
from common.xattr import setxattr as setattr1

cached_attributes = {}
def getxattr(path, attr_name):
  if (path, attr_name) not in cached_attributes:
    response = getattr1(path, attr_name)
    cached_attributes[(path, attr_name)] = response
  return cached_attributes[(path, attr_name)]

def setxattr(path, attr_name, attr_value):
  cached_attributes.pop((path, attr_name), None)
  return setattr1(path, attr_name, attr_value)
