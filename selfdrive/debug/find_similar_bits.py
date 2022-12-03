import os
os.environ["FILEREADER_CACHE"] = "1"
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from tools.lib.route import Route
from tools.lib.logreader import LogReader
from tools.lib.logreader import MultiLogIterator
from opendbc.can.parser import CANParser


# cp = CANParser("Model3CAN", [
#   # ('SteeringSpeed129', 'ID129SteeringAngle'),
# ], checks=[
#   # ('ID129SteeringAngle', 10),
# ], bus=6)

# Route, start and end segment
route = Route("f15e3c37c118e841|2022-12-02--03-32-45"), 4, 6  # camry find pcm fault signal
lr = MultiLogIterator(route[0].log_paths()[route[1]:route[2]])
all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

SEARCH_BUS = 0
MIN_MSGS = 100

bit_to_find = 0
# mismatches = {}
mismatches = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
total_msgs = defaultdict(int)
enabled = False
enabled_t = 0

i = 0
for msg in tqdm(all_msgs):
  if msg.which() == 'pandaStates':
    if msg.pandaStates[0].safetyTxBlocked != 3:
      bit_to_find = 1
  elif msg.which() == 'carState':
    if not enabled and msg.carState.cruiseState.enabled:
      enabled_t = msg.logMonoTime
  elif msg.which() == 'can':
    # cp.update_string(msg.as_builder().to_bytes())
    # if not cp.can_valid:
    #   continue
    # bit_to_find = abs(cp.vl['ID129SteeringAngle']['SteeringSpeed129']) > 4

    for m in msg.can:
      if m.src != SEARCH_BUS:
        continue

      if m.address in mismatches and len(m.dat) != len(mismatches[m.address]):
        print(f"WARNING: {hex(m.address)} changed length from {len(mismatches[m.address])} to {len(m.dat)}. Skipping...")
        continue

      total_msgs[m.address] += 1

      for _y, byt in enumerate(m.dat):
        for _x in range(8):
          # TODO: this script is in the convention of reading binary left to right, switch?
          flipped_idx = 7 - _x
          bit = (byt & (1 << flipped_idx)) >> flipped_idx
          mismatches[m.address][_y][_x] += bit_to_find != bit

mismatches_by_count = {}
print('Mismatches:')
for msg in mismatches:
  for byt_idx, byt in mismatches[msg].items():
    for bit_idx, bit_mismatches in byt.items():
      if total_msgs[msg] > MIN_MSGS:
        perc_mismatched = round(bit_mismatches / total_msgs[msg] * 100, 2)
        if perc_mismatched < 50:
          mismatches_by_count[f'{hex(msg)=}, bit_mismatches={bit_mismatches} of {total_msgs[msg]} ({perc_mismatched}%), {byt_idx=}, {bit_idx=}'] = perc_mismatched
          # print(f'{hex(msg)=}, bit_mismatches={bit_mismatches} of {total_msgs[msg]}, {byt_idx=}, {bit_idx=}')

mismatches_sorted = sorted(mismatches_by_count, key=lambda msg: mismatches_by_count[msg], reverse=True)
for msg in mismatches_sorted:
  print(msg)

print(f'Searched bus: {SEARCH_BUS}')
