"""Install exception handler for process crash."""
import os
import sys

if os.getenv("NOLOG"):
  def capture_exception(*exc_info):
    pass
  def install():
    pass
else:
  from raven import Client
  from raven.transport.http import HTTPTransport

  client = Client('https://1994756b5e6f41cf939a4c65de45f4f2:cefebaf3a8aa40d182609785f7189bd7@app.getsentry.com/77924',
                  install_sys_hook=False, transport=HTTPTransport)

  capture_exception = client.captureException

  def install():
    # installs a sys.excepthook
    __excepthook__ = sys.excepthook
    def handle_exception(*exc_info):
      if exc_info[0] not in (KeyboardInterrupt, SystemExit):
        client.captureException(exc_info=exc_info)
      __excepthook__(*exc_info)
    sys.excepthook = handle_exception
