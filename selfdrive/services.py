import os
import yaml

class Service(object):
  def __init__(self, port, should_log):
    self.port = port
    self.should_log = should_log

service_list_path = os.path.join(os.path.dirname(__file__), "service_list.yaml")

service_list = {}
with open(service_list_path, "r") as f:
  for k, v in yaml.load(f).iteritems():
    service_list[k] = Service(v[0], v[1])
