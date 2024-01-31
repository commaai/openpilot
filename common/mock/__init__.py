"""
Utilities for generating mock messages for testing.
example in common/tests/test_mock.py
"""


import functools
import threading
import time
from typing import List, Union
from cereal.messaging import PubMaster
from cereal.services import SERVICE_LIST
from openpilot.common.mock.generators import generate_liveLocationKalman


MOCK_GENERATOR = {
  "liveLocationKalman": generate_liveLocationKalman
}


def generate_messages_loop(services: List[str], done: threading.Event):
  pm = PubMaster(services)
  i = 0
  while not done.is_set():
    for s in services:
      if i % 100 == SERVICE_LIST[s].frequency:
        message = MOCK_GENERATOR[s]()
        pm.send(s, message)
    i += 1
    time.sleep(1/100)


def mock_messages(services: Union[List[str], str]):
  if isinstance(services, str):
    services = [services]

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      done = threading.Event()
      t = threading.Thread(target=generate_messages_loop, args=(services, done))
      t.start()
      ret = func(*args, **kwargs)
      done.set()
      t.join()
      return ret
    return wrapper
  return decorator
