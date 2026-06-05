import os

from openpilot.common.basedir import BASEDIR
from opendbc.car.docs import generate_cars_md, get_all_car_docs
from openpilot.selfdrive.debug.dump_car_docs import dump_car_docs
from openpilot.selfdrive.debug.print_docs_diff import print_car_docs_diff
from openpilot.selfdrive.car.docs import CARS_MD_TEMPLATE


class TestCarDocs:
  @classmethod
  def setup_class(cls):
    cls.all_cars = get_all_car_docs()

  def test_generator(self):
    generate_cars_md(self.all_cars, CARS_MD_TEMPLATE)

  def test_docs_diff(self):
    dump_path = os.path.join(BASEDIR, "selfdrive", "car", "tests", "cars_dump")
    dump_car_docs(dump_path)
    print_car_docs_diff(dump_path)
    os.remove(dump_path)
