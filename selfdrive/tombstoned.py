import os
import re
import time
import datetime

from raven import Client
from raven.transport.http import HTTPTransport

from selfdrive.version import version, dirty
from selfdrive.swaglog import cloudlog

def get_tombstones():
  DIR_DATA = "/data/tombstones/"
  return [(DIR_DATA + fn, int(os.stat(DIR_DATA + fn).st_ctime) )
          for fn in os.listdir(DIR_DATA) if fn.startswith("tombstone")]

def report_tombstone(fn, client):
  mtime = os.path.getmtime(fn)
  with open(fn, encoding='ISO-8859-1') as f:
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
  else:
    parsedict = {}

  thread_line = parsedict.get('thread', '')
  thread_parsed = re.match(r'pid: (?P<pid>\d+), tid: (?P<tid>\d+), name: (?P<name>.*) >>> (?P<cmd>.*) <<<', thread_line)
  if thread_parsed:
    thread_parseddict = thread_parsed.groupdict()
  else:
    thread_parseddict = {}
  pid = thread_parseddict.get('pid', '')
  tid = thread_parseddict.get('tid', '')
  name = thread_parseddict.get('name', 'unknown')
  cmd = thread_parseddict.get('cmd', 'unknown')

  signal_line = parsedict.get('signal', '')
  signal_parsed = re.match(r'signal (?P<signal>.*?), code (?P<code>.*?), fault addr (?P<fault_addr>.*)\n', signal_line)
  if signal_parsed:
    signal_parseddict = signal_parsed.groupdict()
  else:
    signal_parseddict = {}
  signal = signal_parseddict.get('signal', 'unknown')
  code = signal_parseddict.get('code', 'unknown')
  fault_addr = signal_parseddict.get('fault_addr', '')

  abort_line = parsedict.get('abort', '')

  if parsed:
    message = 'Process {} ({}) got signal {} code {}'.format(name, cmd, signal, code)
    if abort_line:
      message += '\n'+abort_line
  else:
    message = fn+'\n'+dat[:1024]


  client.captureMessage(
    message=message,
    date=datetime.datetime.utcfromtimestamp(mtime),
    data={
      'logger':'tombstoned',
      'platform':'other',
    },
    sdk={'name': 'tombstoned', 'version': '0'},
    extra={
      'fault_addr': fault_addr,
      'abort_msg': abort_line,
      'pid': pid,
      'tid': tid,
      'name':'{} ({})'.format(name, cmd),
      'tombstone_fn': fn,
      'header': parsedict.get('header'),
      'registers': parsedict.get('registers'),
      'backtrace': parsedict.get('backtrace'),
      'logtail': logtail,
    },
    tags={
      'name':'{} ({})'.format(name, cmd),
      'signal':signal,
      'code':code,
      'fault_addr':fault_addr,
    },
  )
  cloudlog.error({'tombstone': message})


def main(gctx=None):
  initial_tombstones = set(get_tombstones())

  client = Client('https://d3b175702f62402c91ade04d1c547e68:b20d68c813c74f63a7cdf9c4039d8f56@sentry.io/157615',
                  install_sys_hook=False, transport=HTTPTransport, release=version, tags={'dirty': dirty}, string_max_length=10000)

  client.user_context({'id': os.environ.get('DONGLE_ID')})
  while True:
    now_tombstones = set(get_tombstones())

    for fn, ctime in (now_tombstones - initial_tombstones):
      cloudlog.info("reporting new tombstone %s", fn)
      report_tombstone(fn, client)

    initial_tombstones = now_tombstones
    time.sleep(5)

if __name__ == "__main__":
  main()
