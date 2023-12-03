#!/usr/bin/env python3
import unittest
import pytest
import sys

from parameterized import parameterized_class
from typing import List, Optional

from openpilot.selfdrive.car.car_helpers import interface_names
from openpilot.selfdrive.test.process_replay.process_replay import check_openpilot_enabled
from openpilot.selfdrive.test.process_replay.helpers import TestProcessReplayDiffBase


source_segments = [
  ("BODY", "937ccb7243511b65|2022-05-24--16-03-09--1"),        # COMMA.BODY
  ("HYUNDAI", "02c45f73a2e5c6e9|2021-01-01--19-08-22--1"),     # HYUNDAI.SONATA
  ("HYUNDAI2", "d545129f3ca90f28|2022-11-07--20-43-08--3"),    # HYUNDAI.KIA_EV6 (+ QCOM GPS)
  ("TOYOTA", "0982d79ebb0de295|2021-01-04--17-13-21--13"),     # TOYOTA.PRIUS
  ("TOYOTA2", "0982d79ebb0de295|2021-01-03--20-03-36--6"),     # TOYOTA.RAV4
  ("TOYOTA3", "f7d7e3538cda1a2a|2021-08-16--08-55-34--6"),     # TOYOTA.COROLLA_TSS2
  ("HONDA", "eb140f119469d9ab|2021-06-12--10-46-24--27"),      # HONDA.CIVIC (NIDEC)
  ("HONDA2", "7d2244f34d1bbcda|2021-06-25--12-25-37--26"),     # HONDA.ACCORD (BOSCH)
  ("CHRYSLER", "4deb27de11bee626|2021-02-20--11-28-55--8"),    # CHRYSLER.PACIFICA_2018_HYBRID
  ("RAM", "17fc16d840fe9d21|2023-04-26--13-28-44--5"),         # CHRYSLER.RAM_1500
  ("SUBARU", "341dccd5359e3c97|2022-09-12--10-35-33--3"),      # SUBARU.OUTBACK
  ("GM", "0c58b6a25109da2b|2021-02-23--16-35-50--11"),         # GM.VOLT
  ("GM2", "376bf99325883932|2022-10-27--13-41-22--1"),         # GM.BOLT_EUV
  ("NISSAN", "35336926920f3571|2021-02-12--18-38-48--46"),     # NISSAN.XTRAIL
  ("VOLKSWAGEN", "de9592456ad7d144|2021-06-29--11-00-15--6"),  # VOLKSWAGEN.GOLF
  ("MAZDA", "bd6a637565e91581|2021-10-30--15-14-53--4"),       # MAZDA.CX9_2021
  ("FORD", "54827bf84c38b14f|2023-01-26--21-59-07--4"),        # FORD.BRONCO_SPORT_MK1

  # Enable when port is tested and dashcamOnly is no longer set
  #("TESLA", "bb50caf5f0945ab1|2021-06-19--17-20-18--3"),      # TESLA.AP2_MODELS
  #("VOLKSWAGEN2", "3cfdec54aa035f3f|2022-07-19--23-45-10--2"),  # VOLKSWAGEN.PASSAT_NMS
]

segments = [
  ("BODY", "regen997DF2697CB|2023-10-30--23-14-29--0"),
  ("HYUNDAI", "regen2A9D2A8E0B4|2023-10-30--23-13-34--0"),
  ("HYUNDAI2", "regen6CA24BC3035|2023-10-30--23-14-28--0"),
  ("TOYOTA", "regen5C019D76307|2023-10-30--23-13-31--0"),
  ("TOYOTA2", "regen5DCADA88A96|2023-10-30--23-14-57--0"),
  ("TOYOTA3", "regen7204CA3A498|2023-10-30--23-15-55--0"),
  ("HONDA", "regen048F8FA0B24|2023-10-30--23-15-53--0"),
  ("HONDA2", "regen7D2D3F82D5B|2023-10-30--23-15-55--0"),
  ("CHRYSLER", "regen7125C42780C|2023-10-30--23-16-21--0"),
  ("RAM", "regen2731F3213D2|2023-10-30--23-18-11--0"),
  ("SUBARU", "regen86E4C1B4DDD|2023-10-30--23-18-14--0"),
  ("GM", "regenF6393D64745|2023-10-30--23-17-18--0"),
  ("GM2", "regen220F830C05B|2023-10-30--23-18-39--0"),
  ("NISSAN", "regen4F671F7C435|2023-10-30--23-18-40--0"),
  ("VOLKSWAGEN", "regen8BDFE7307A0|2023-10-30--23-19-36--0"),
  ("MAZDA", "regen2E9F1A15FD5|2023-10-30--23-20-36--0"),
  ("FORD", "regen6D39E54606E|2023-10-30--23-20-54--0"),
  ]

# dashcamOnly makes don't need to be tested until a full port is done
excluded_interfaces = ["mock", "tesla"]

ALL_CARS = sorted({car for car, _ in segments})


@pytest.mark.slow
@parameterized_class(('case_name', 'segment'), segments)
@pytest.mark.xdist_group_class_property('case_name')
class TestCarProcessReplay(TestProcessReplayDiffBase):
  """
  Runs a replay diff on a segment for each car.
  """

  case_name: Optional[str] = None
  tested_cars: List[str] = ALL_CARS

  @classmethod
  def setUpClass(cls):
    if cls.case_name not in cls.tested_cars:
      raise unittest.SkipTest(f"{cls.case_name} was not requested to be tested")
    super().setUpClass()

  def test_all_makes_are_tested(self):
    if set(self.tested_cars) != set(ALL_CARS):
      raise unittest.SkipTest("skipping check because some cars were skipped via command line")

    # check to make sure all car brands are tested
    untested = (set(interface_names) - set(excluded_interfaces)) - {c.lower() for c in self.tested_cars}
    self.assertEqual(len(untested), 0, f"Cars missing routes: {str(untested)}")

  def test_controlsd_engaged(self):
    if "controlsd" not in self.tested_procs:
      raise unittest.SkipTest("controlsd was not requested to be tested")

    # check to make sure openpilot is engaged in the route
    log_msgs = self.log_msgs["controlsd"]
    self.assertTrue(check_openpilot_enabled(log_msgs), f"Route did not enable at all or for long enough: {self.segment}")


if __name__ == '__main__':
  pytest.main([*sys.argv[1:], __file__])