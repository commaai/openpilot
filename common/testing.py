import os
from nose.tools import nottest

def phone_only(x):
  if os.path.isfile("/init.qcom.rc"):
    return x
  else:
    return nottest(x)
