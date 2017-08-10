import os
import sys
import copy
import json
import socket
import logging
from threading import local
from collections import OrderedDict
from contextlib import contextmanager

def json_handler(obj):
  # if isinstance(obj, (datetime.date, datetime.time)):
  #   return obj.isoformat()
  return repr(obj)

def json_robust_dumps(obj):
  return json.dumps(obj, default=json_handler)

class NiceOrderedDict(OrderedDict):
  def __str__(self):
    return '{'+', '.join("%r: %r" % p for p in self.iteritems())+'}'

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
    return json_robust_dumps(self.format_dict(record))

_tmpfunc = lambda: 0
_srcfile = os.path.normcase(_tmpfunc.__code__.co_filename)

class SwagLogger(logging.Logger):
  def __init__(self):
    logging.Logger.__init__(self, "swaglog")

    self.global_ctx = {}

    self.log_local = local()
    self.log_local.ctx = {}

  def findCaller(self):
    """
      Find the stack frame of the caller so that we can note the source
      file name, line number and function name.
      """
    # f = currentframe()
    f = sys._getframe(3)
    #On some versions of IronPython, currentframe() returns None if
    #IronPython isn't run with -X:Frames.
    if f is not None:
      f = f.f_back
    rv = "(unknown file)", 0, "(unknown function)"
    while hasattr(f, "f_code"):
      co = f.f_code
      filename = os.path.normcase(co.co_filename)
      if filename in (logging._srcfile, _srcfile):
        f = f.f_back
        continue
      rv = (co.co_filename, f.f_lineno, co.co_name)
      break
    return rv

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

  def event(self, event_name, *args, **kwargs):
    evt = NiceOrderedDict()
    evt['event'] = event_name
    if args:
      evt['args'] = args
    evt.update(kwargs)
    self.info(evt)

if __name__ == "__main__":
  log = SwagLogger()

  log.info("asdasd %s", "a")
  log.info({'wut': 1})

  with log.ctx():
    log.bind(user="some user")
    log.info("in req")
    log.event("do_req")
