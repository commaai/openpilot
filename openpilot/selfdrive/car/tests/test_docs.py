from opendbc.car.docs import generate_cars_md, get_all_car_docs
from openpilot.selfdrive.car.docs import CARS_MD_TEMPLATE


class TestCarDocs:
  @classmethod
  def setup_class(cls):
    cls.all_cars = get_all_car_docs()

  def test_generator(self):
    generate_cars_md(self.all_cars, CARS_MD_TEMPLATE)
