from collections import defaultdict
from selfdrive.car.honda.values import FW_VERSIONS, HONDA_BOSCH


NEW_FW_VERSIONS = {}  # defaultdict(dict)

BOSCH_ADDRS = (0x18DAB0F1, 0x18DAB5F1)
NIDEC_ADDRS = (0x18DA30F1, 0x18DA53F1, 0x18DAB0F1)

for car, fw_by_ecu in FW_VERSIONS.items():
  NEW_FW_VERSIONS[car] = dict()
  for ecu_type, fws in fw_by_ecu.items():
    if car in HONDA_BOSCH:
      if ecu_type[1] in BOSCH_ADDRS:
        NEW_FW_VERSIONS[car][ecu_type] = fws
    else:
      if ecu_type[1] in NIDEC_ADDRS:
        NEW_FW_VERSIONS[car][ecu_type] = fws

print(NEW_FW_VERSIONS)
