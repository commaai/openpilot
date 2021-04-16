"""Install exception handler for process crash."""
import os
import sys
import capnp
from selfdrive.version import version, dirty, origin, branch

from selfdrive.hardware import PC
from selfdrive.swaglog import cloudlog

if os.getenv("NOLOG") or os.getenv("NOCRASH") or PC:
  def capture_exception(*args, **kwargs):
    pass

  def bind_user(**kwargs):
    pass

  def bind_extra(**kwargs):
    pass

else:
  from raven import Client
  from raven.transport.http import HTTPTransport

  tags = {
    'dirty': dirty,
    'origin': origin,
    'branch': branch
  }
  client = Client('https://a8dc76b5bfb34908a601d67e2aa8bcf9:4336ee4648984e438370a3fa3f5adda2@o33823.ingest.sentry.io/77924',
                  install_sys_hook=False, transport=HTTPTransport, release=version, tags=tags)

  def capture_exception(*args, **kwargs):
    exc_info = sys.exc_info()
    if not exc_info[0] is capnp.lib.capnp.KjException:
      client.captureException(*args, **kwargs)
    cloudlog.error("crash", exc_info=kwargs.get('exc_info', 1))

  def bind_user(**kwargs):
    client.user_context(kwargs)

  def bind_extra(**kwargs):
    client.extra_context(kwargs)

