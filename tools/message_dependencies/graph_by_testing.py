from itertools import combinations

from selfdrive.manager.process_config import managed_processes
from cereal import log

msgs = list(log.Event.schema.fieldnames)

exit()
for proc in managed_processes:
  proc.prepare()
  proc.start()
  # assume no proccess needs more than 5 inputs
  for i in range(1, 5):
    for msg_comb in combinations(msg, i):
      for msg in msg_comb:
        data = log.Event.new_message().as_builder()
        pm.send(msg, log.Event.new_message())



