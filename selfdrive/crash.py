"""Install exception handler for process crash."""
from selfdrive.swaglog import cloudlog
from selfdrive.version import version

import sentry_sdk
from sentry_sdk.integrations.threading import ThreadingIntegration

def capture_exception(*args, **kwargs) -> None:
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

def init() -> None:
  sentry_sdk.init("https://980a0cba712a4c3593c33c78a12446e1:fecab286bcaf4dba8b04f7cff0188e2d@sentry.io/1488600",
                  default_integrations=False, integrations=[ThreadingIntegration(propagate_hub=True)],
                  release=version)
