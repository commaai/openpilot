import os

from openpilot.common.basedir import BASEDIR
from opendbc.car.docs.generate_docs import generate_cars_md, get_all_car_docs, CARS_MD_OUT, CARS_MD_TEMPLATE
from opendbc.car.docs.dump_car_docs import dump_car_docs
from opendbc.car.docs.print_docs_diff import print_car_docs_diff


class TestCarDocs:
  @classmethod
  def setup_class(cls):
    cls.all_cars = get_all_car_docs()

  def test_generator(self):
    generated_cars_md = generate_cars_md(self.all_cars, CARS_MD_TEMPLATE)
    with open(CARS_MD_OUT) as f:
      current_cars_md = f.read()

    assert generated_cars_md == current_cars_md, "Run selfdrive/opcar/docs.py to update the compatibility documentation"

  def test_docs_diff(self):
    dump_path = os.path.join(BASEDIR, "selfdrive", "car", "tests", "cars_dump")
    dump_car_docs(dump_path)
    print_car_docs_diff(dump_path)
    os.remove(dump_path)
