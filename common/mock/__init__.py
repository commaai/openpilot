"""
Utilities for generating mock messages for testing.
example in common/tests/test_mock.py
"""


import functools
import threading
from cereal.messaging import PubMaster
from cereal.services import SERVICE_LIST
from openpilot.common.mock.generators import generate_liveLocationKalman
from openpilot.common.realtime import Ratekeeper


MOCK_GENERATOR = {
  "liveLocationKalman": generate_liveLocationKalman
}


def generate_messages_loop(services: list[str], done: threading.Event):
  pm = PubMaster(services)
  rk = Ratekeeper(100)
  i = 0
  while not done.is_set():
    for s in services:
      should_send = i % (100/SERVICE_LIST[s].frequency) == 0
      if should_send:
        message = MOCK_GENERATOR[s]()
        pm.send(s, message)
    i += 1
    rk.keep_time()


def mock_messages(services: list[str] | str):
  if isinstance(services, str):
    services = [services]

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      done = threading.Event()
      t = threading.Thread(target=generate_messages_loop, args=(services, done))
      t.start()
      try:
        return func(*args, **kwargs)
      finally:
        done.set()
        t.join()
    return wrapper
  return decorator
