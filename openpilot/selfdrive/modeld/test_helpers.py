from types import SimpleNamespace

import pytest

from openpilot.selfdrive.modeld.helpers import chestnut_ready


@pytest.mark.parametrize(("voltage", "fault", "ltssm", "expected"), [
  (12000, False, 0x78, True),
  (4999, False, 0x78, False),
  (12000, True, 0x78, False),
  (12000, False, 0, False),
])
def test_chestnut_hardware_ready(voltage, fault, ltssm, expected):
  state = SimpleNamespace(supplyVoltage=voltage, supplyFault=fault, pcieLtssm=ltssm)
  assert chestnut_ready(state) == expected
