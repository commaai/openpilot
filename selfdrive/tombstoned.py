import os
import re
import time
import uuid
import datetime

from raven import Client
from raven.transport.http import HTTPTransport

from selfdrive.version import version
from selfdrive.swaglog import cloudlog

def get_tombstones():
  return [fn for fn in os.listdir("/data/tombstones") if fn.startswith("tombstone")]

def report_tombstone(fn, client):
  mtime = os.path.getmtime(fn)
  with open(fn, "r") as f:
    dat = f.read()

  # see system/core/debuggerd/tombstone.cpp
  parsed = re.match(r"[* ]*\n"
                    r"(?P<header>CM Version:[\s\S]*?ABI:.*\n)"
                    r"(?P<thread>pid:.*\n)"
                    r"(?P<signal>signal.*\n)?"
                    r"(?P<abort>Abort.*\n)?"
                    r"(?P<registers>\s+x0[\s\S]*?\n)\n"
                    r"(?:backtrace:\n"
                      r"(?P<backtrace>[\s\S]*?\n)\n"
                      r"stack:\n"
                      r"(?P<stack>[\s\S]*?\n)\n"
                    r")?", dat)

  logtail = re.search(r"--------- tail end of.*\n([\s\S]*?\n)---", dat)
  logtail = logtail and logtail.group(1)

  if parsed:
    parsedict = parsed.groupdict()
    message = parsedict.get('thread') or ''
    message += parsedict.get('signal') or  ''
    message += parsedict.get('abort') or ''
  else:
    parsedict = {}
    message = fn+"\n"+dat[:1024]

  client.send(
    event_id=uuid.uuid4().hex,
    timestamp=datetime.datetime.utcfromtimestamp(mtime),
    logger='tombstoned',
    platform='other',
    sdk={'name': 'tombstoned', 'version': '0'},
    extra={
      'tombstone_fn': fn,
      'header': parsedict.get('header'),
      'registers': parsedict.get('registers'),
      'backtrace': parsedict.get('backtrace'),
      'logtail': logtail,
      'version': version,
      'dirty': not bool(os.environ.get('CLEAN')),
    },
    user={'id': os.environ.get('DONGLE_ID')},
    message=message,
  )


def main(gctx):
  initial_tombstones = set(get_tombstones())

  client = Client('https://d3b175702f62402c91ade04d1c547e68:b20d68c813c74f63a7cdf9c4039d8f56@sentry.io/157615',
                  install_sys_hook=False, transport=HTTPTransport)

  while True:
    now_tombstones = set(get_tombstones())

    for ts in (now_tombstones - initial_tombstones):
      fn = "/data/tombstones/"+ts
      cloudlog.info("reporting new tombstone %s", fn)
      report_tombstone(fn, client)

    initial_tombstones = now_tombstones
    time.sleep(5)

if __name__ == "__main__":
  main(None)
