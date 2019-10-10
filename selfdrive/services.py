import os
import yaml

class Service():
  def __init__(self, port, should_log, frequency, decimation=None):
    self.port = port
    self.should_log = should_log
    self.frequency = frequency
    self.decimation = decimation

service_list_path = os.path.join(os.path.dirname(__file__), "service_list.yaml")

service_list = {}
with open(service_list_path, "r") as f:
  for k, v in yaml.safe_load(f).items():
    decimation = None
    if len(v) == 4:
      decimation = v[3]

    service_list[k] = Service(v[0], v[1], v[2], decimation)
