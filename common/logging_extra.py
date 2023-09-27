import io
import os
import sys
import copy
import json
import time
import uuid
import socket
import logging
import traceback
from threading import local
from collections import OrderedDict
from contextlib import contextmanager

LOG_TIMESTAMPS = "LOG_TIMESTAMPS" in os.environ

def json_handler(obj):
  # if isinstance(obj, (datetime.date, datetime.time)):
  #   return obj.isoformat()
  return repr(obj)

def json_robust_dumps(obj):
  return json.dumps(obj, default=json_handler)

class NiceOrderedDict(OrderedDict):
  def __str__(self):
    return json_robust_dumps(self)

class SwagFormatter(logging.Formatter):
  def __init__(self, swaglogger):
    logging.Formatter.__init__(self, None, '%a %b %d %H:%M:%S %Z %Y')

    self.swaglogger = swaglogger
    self.host = socket.gethostname()

  def format_dict(self, record):
    record_dict = NiceOrderedDict()

    if isinstance(record.msg, dict):
      record_dict['msg'] = record.msg
    else:
      try:
        record_dict['msg'] = record.getMessage()
      except (ValueError, TypeError):
        record_dict['msg'] = [record.msg]+record.args

    record_dict['ctx'] = self.swaglogger.get_ctx()

    if record.exc_info:
      record_dict['exc_info'] = self.formatException(record.exc_info)

    record_dict['level'] = record.levelname
    record_dict['levelnum'] = record.levelno
    record_dict['name'] = record.name
    record_dict['filename'] = record.filename
    record_dict['lineno'] = record.lineno
    record_dict['pathname'] = record.pathname
    record_dict['module'] = record.module
    record_dict['funcName'] = record.funcName
    record_dict['host'] = self.host
    record_dict['process'] = record.process
    record_dict['thread'] = record.thread
    record_dict['threadName'] = record.threadName
    record_dict['created'] = record.created

    return record_dict

  def format(self, record):
    if self.swaglogger is None:
      raise Exception("must set swaglogger before calling format()")
    return json_robust_dumps(self.format_dict(record))

class SwagLogFileFormatter(SwagFormatter):
  def fix_kv(self, k, v):
    # append type to names to preserve legacy naming in logs
    # avoids overlapping key namespaces with different types
    # e.g. log.info() creates 'msg' -> 'msg$s'
    #      log.event() creates 'msg.health.logMonoTime' -> 'msg.health.logMonoTime$i'
    #      because overlapping namespace 'msg' caused problems
    if isinstance(v, (str, bytes)):
      k += "$s"
    elif isinstance(v, float):
      k += "$f"
    elif isinstance(v, bool):
      k += "$b"
    elif isinstance(v, int):
      k += "$i"
    elif isinstance(v, dict):
      nv = {}
      for ik, iv in v.items():
        ik, iv = self.fix_kv(ik, iv)
        nv[ik] = iv
      v = nv
    elif isinstance(v, list):
      k += "$a"
    return k, v

  def format(self, record):
    if isinstance(record, str):
      v = json.loads(record)
    else:
      v = self.format_dict(record)

    mk, mv = self.fix_kv('msg', v['msg'])
    del v['msg']
    v[mk] = mv
    v['id'] = uuid.uuid4().hex

    return json_robust_dumps(v)

class SwagErrorFilter(logging.Filter):
  def filter(self, record):
    return record.levelno < logging.ERROR

def _tmpfunc():
  return 0

def _srcfile():
  return os.path.normcase(_tmpfunc.__code__.co_filename)

class SwagLogger(logging.Logger):
  def __init__(self):
    logging.Logger.__init__(self, "swaglog")

    self.global_ctx = {}

    self.log_local = local()
    self.log_local.ctx = {}

  def local_ctx(self):
    try:
      return self.log_local.ctx
    except AttributeError:
      self.log_local.ctx = {}
      return self.log_local.ctx

  def get_ctx(self):
    return dict(self.local_ctx(), **self.global_ctx)

  @contextmanager
  def ctx(self, **kwargs):
    old_ctx = self.local_ctx()
    self.log_local.ctx = copy.copy(old_ctx) or {}
    self.log_local.ctx.update(kwargs)
    try:
      yield
    finally:
      self.log_local.ctx = old_ctx

  def bind(self, **kwargs):
    self.local_ctx().update(kwargs)

  def bind_global(self, **kwargs):
    self.global_ctx.update(kwargs)

  def event(self, event, *args, **kwargs):
    evt = NiceOrderedDict()
    evt['event'] = event
    if args:
      evt['args'] = args
    evt.update(kwargs)
    if 'error' in kwargs:
      self.error(evt)
    elif 'debug' in kwargs:
      self.debug(evt)
    else:
      self.info(evt)

  def timestamp(self, event_name):
    if LOG_TIMESTAMPS:
      t = time.monotonic()
      tstp = NiceOrderedDict()
      tstp['timestamp'] = NiceOrderedDict()
      tstp['timestamp']["event"] = event_name
      tstp['timestamp']["time"] = t*1e9
      self.debug(tstp)

  def findCaller(self, stack_info=False, stacklevel=1):
    """
    Find the stack frame of the caller so that we can note the source
    file name, line number and function name.
    """
    f = sys._getframe(3)
    #On some versions of IronPython, currentframe() returns None if
    #IronPython isn't run with -X:Frames.
    if f is not None:
      f = f.f_back
    orig_f = f
    while f and stacklevel > 1:
      f = f.f_back
      stacklevel -= 1
    if not f:
      f = orig_f
    rv = "(unknown file)", 0, "(unknown function)", None
    while hasattr(f, "f_code"):
      co = f.f_code
      filename = os.path.normcase(co.co_filename)

      # TODO: is this pylint exception correct?
      if filename == _srcfile:  # pylint: disable=comparison-with-callable
        f = f.f_back
        continue
      sinfo = None
      if stack_info:
        sio = io.StringIO()
        sio.write('Stack (most recent call last):\n')
        traceback.print_stack(f, file=sio)
        sinfo = sio.getvalue()
        if sinfo[-1] == '\n':
          sinfo = sinfo[:-1]
        sio.close()
      rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
      break
    return rv

if __name__ == "__main__":
  log = SwagLogger()

  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.INFO)
  stdout_handler.addFilter(SwagErrorFilter())
  log.addHandler(stdout_handler)

  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setLevel(logging.ERROR)
  log.addHandler(stderr_handler)

  log.info("asdasd %s", "a")
  log.info({'wut': 1})
  log.warning("warning")
  log.error("error")
  log.critical("critical")
  log.event("test", x="y")

  with log.ctx():
    stdout_handler.setFormatter(SwagFormatter(log))
    stderr_handler.setFormatter(SwagFormatter(log))
    log.bind(user="some user")
    log.info("in req")
    print("")
    log.warning("warning")
    print("")
    log.error("error")
    print("")
    log.critical("critical")
    print("")
    log.event("do_req", a=1, b="c")
