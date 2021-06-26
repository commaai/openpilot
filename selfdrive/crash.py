"""Install exception handler for process crash."""
from selfdrive.swaglog import cloudlog
from selfdrive.version import version
import traceback
from datetime import datetime

import sentry_sdk
from sentry_sdk.integrations.threading import ThreadingIntegration

def save_exception(exc_text):
  COMMUNITY_DIR = '/data/community'
  CRASHES_DIR = '{}/crashes'.format(COMMUNITY_DIR)

  if not os.path.exists(COMMUNITY_DIR):
    os.mkdir(COMMUNITY_DIR)
  if not os.path.exists(CRASHES_DIR):
    os.mkdir(CRASHES_DIR)
  i = 0
  log_file = '{}/{}'.format(CRASHES_DIR, datetime.now().strftime('%Y-%m-%d--%H-%M-%S.%f.log')[:-3])
  if os.path.exists(log_file):
    while os.path.exists(log_file + str(i)):
      i += 1
    log_file += str(i)
  with open(log_file, 'w') as f:
    f.write(exc_text)
  print('Logged current crash to {}'.format(log_file))

def capture_exception(*args, **kwargs):
  cloudlog.error("crash", exc_info=kwargs.get('exc_info', 1))

  try:
    sentry_sdk.capture_exception(*args, **kwargs)
    sentry_sdk.flush()  # https://github.com/getsentry/sentry-python/issues/291
  except Exception:
    cloudlog.exception("sentry exception")

def bind_user(**kwargs) -> None:
  sentry_sdk.set_user(kwargs)

def bind_extra(**kwargs) -> None:
  for k, v in kwargs.items():
    sentry_sdk.set_tag(k, v)

def init():
  sentry_sdk.init("https://137e8e621f114f858f4c392c52e18c6d:8aba82f49af040c8aac45e95a8484970@sentry.io/1404547",
                  default_integrations=False, integrations=[ThreadingIntegration(propagate_hub=True)],
                  release=version)
