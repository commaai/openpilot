import logging

from common.compat import basestring
from common.logging_extra import SwagFormatter, json_robust_dumps

from logstash_async.handler import AsynchronousLogstashHandler
from logstash_async.transport import UdpTransport

# elasticsearch doesnt allow different 'keys' with the different types in the same index

class SwagLogstashFormatter(SwagFormatter):
  def __init__(self, swaglogger):
    super(SwagLogstashFormatter, self).__init__(swaglogger)

  def fix_kv(self, k, v):
    if isinstance(v, basestring):
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
    v = self.format_dict(record)

    mk, mv = self.fix_kv('msg', v['msg'])
    del v['msg']
    v[mk] = mv

    return json_robust_dumps(v)

class SwagLogstashHandler(AsynchronousLogstashHandler):
  def __init__(self, host, port, name, formatter):
    super(SwagLogstashHandler, self).__init__(host, port, database_path=None, transport=UdpTransport)
    self.name = name
    if not isinstance(formatter, SwagLogstashFormatter):
      raise ValueError("formatter must be swag")
    self.setFormatter(formatter)

  def emit(self, record):
    record.name = self.name
    super(SwagLogstashHandler, self).emit(record)

if __name__ == "__main__":
  from common.logging_extra import SwagLogger
  log = SwagLogger()
  ls_formatter = SwagLogstashFormatter(log)
  ls_handler = SwagLogstashHandler("logstash.comma.life", 5040, "pipeline", ls_formatter)
  log.addHandler(ls_handler)
  s_handler = logging.StreamHandler()
  log.addHandler(s_handler)

  log.info("asynclogtest %s", "1")
  log.info({'asynclogtest': 2})
  log.warning("asynclogtest warning")
  log.error("asynclogtest error")
  log.critical("asynclogtest critical")
  log.event("asynclogtest", a="b")
