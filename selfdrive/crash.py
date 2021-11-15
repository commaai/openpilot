"""Install exception handler for process crash."""
from selfdrive.swaglog import cloudlog
from selfdrive.version import version
from selfdrive.version import origin, branch, dirty, smiskol_remote, get_git_commit
from common.op_params import opParams

import sentry_sdk
from sentry_sdk import set_tag
from sentry_sdk.integrations.threading import ThreadingIntegration

def capture_exception(*args, **kwargs) -> None:
  cloudlog.error("crash", exc_info=kwargs.get('exc_info', 1))

  try:
    sentry_sdk.capture_exception(*args, **kwargs)
    sentry_sdk.flush()  # https://github.com/getsentry/sentry-python/issues/291
  except Exception:
    cloudlog.exception("sentry exception")

def capture_message(*args, **kwargs) -> None:
  try:
    sentry_sdk.capture_message(*args, **kwargs)
    sentry_sdk.flush()  # https://github.com/getsentry/sentry-python/issues/291
  except Exception:
    cloudlog.exception("sentry exception")

def bind_user(**kwargs) -> None:
  sentry_sdk.set_user(kwargs)

def bind_extra(**kwargs) -> None:
  for k, v in kwargs.items():
    sentry_sdk.set_tag(k, v)

def init() -> None:
  sentry_uri = 'https://30d4f5e7d35c4a0d84455c03c0e80706@o237581.ingest.sentry.io/5844043'

  sentry_sdk.init(sentry_uri, default_integrations=False,
                  integrations=[ThreadingIntegration(propagate_hub=True)], release=version)

  error_tags = {'dirty': dirty, 'origin': origin, 'branch': branch, 'commit': get_git_commit(), 'username': opParams().get('username')}
  if error_tags['username'] is None or not isinstance(error_tags['username'], str):
    error_tags['username'] = 'undefined'
  for tag in error_tags:
    set_tag(tag, error_tags[tag])
