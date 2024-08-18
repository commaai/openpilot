from opendbc.car.docs import generate_cars_md, get_all_car_docs
from openpilot.selfdrive.car.docs import CARS_MD_OUT, CARS_MD_TEMPLATE


class TestCarDocs:
  @classmethod
  def setup_class(cls):
    cls.all_cars = get_all_car_docs()

  def test_generator(self):
    generated_cars_md = generate_cars_md(self.all_cars, CARS_MD_TEMPLATE)
    with open(CARS_MD_OUT) as f:
      current_cars_md = f.read()

    assert generated_cars_md == current_cars_md, "Run selfdrive/opcar/docs.py to update the compatibility documentation"
