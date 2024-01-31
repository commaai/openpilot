"""
Utilities for generating mock messages for testing. 
example in common/tests/test_mock.py
"""


import functools
import threading
from typing import List, Union
from cereal.messaging import PubMaster
from openpilot.common.mock.generators import generate_liveLocationKalman


MOCK_GENERATOR = {
  "liveLocationKalman": generate_liveLocationKalman
}


def generate_messages_loop(names: List[str], done: threading.Event):
  pm = PubMaster(names)
  while not done.is_set():
    for name in names:
      message = MOCK_GENERATOR[name]()
      pm.send(name, message)


def mock_messages(names: Union[List[str], str]):
  if isinstance(names, str):
    names = [names]

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      done = threading.Event()
      t = threading.Thread(target=generate_messages_loop, args=(names, done))
      t.start()
      ret = func(*args, **kwargs)
      done.set()
      t.join()
      return ret
    return wrapper
  return decorator