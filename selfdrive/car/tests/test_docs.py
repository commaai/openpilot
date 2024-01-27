#!/usr/bin/env python3
from collections import defaultdict
import os
import re
import unittest

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car.car_helpers import interfaces, get_interface_attr
from openpilot.selfdrive.car.docs import CARS_MD_OUT, CARS_MD_TEMPLATE, generate_cars_md, get_all_car_info
from openpilot.selfdrive.car.docs_definitions import Cable, Column, PartType, Star
from openpilot.selfdrive.car.honda.values import CAR as HONDA
from openpilot.selfdrive.debug.dump_car_info import dump_car_info
from openpilot.selfdrive.debug.print_docs_diff import print_car_info_diff, process_markdown_file, compare_car_info


class TestCarDocs(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.all_cars = get_all_car_info()

    cls.mock_markdown_content = """
        <!-- ALL CAR INFO HERE -->
        |Make|Model|Supported Package|ACC|No ACC accel below|No ALC below|Steering Torque|Resume from stop|<a href="##"><img width=2000></a>Hardware Needed<br>&nbsp;|Video|
        |---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |Acura|ILX 2016-19|AcuraWatch Plus|openpilot|25 mph|25 mph|[![star](assets/icon-star-empty.svg)](##)|[![star](assets/icon-star-empty.svg)](##)|<details><summary>Parts</summary><sub>- 1 Honda Nidec connector<br>- 1 RJ45 cable (7 ft)<br>- 1 comma 3X<br>- 1 comma power v2<br>- 1 harness box<br>- 1 mount<br>- 1 right angle OBD-C cable (1.5 ft)<br><a href="https://comma.ai/shop/comma-3x.html?make=Acura&model=ILX 2016-19">Buy Here</a></sub></details>||<!-- detail sentence:openpilot upgrades your <strong>Acura ILX</strong> with automated lane centering and adaptive cruise control <strong>while driving above 25 mph</strong>. This car may not be able to take tight turns on its own. Traffic light and stop sign handling is also available in <a href='https://blog.comma.ai/090release/#experimental-mode' target='_blank' class='link-light-new-regular-text'>Experimental mode</a>. -->
        <!-- ALL CAR INFO HERE ENDS -->
    """
    cls.mock_file_path = os.path.join(BASEDIR, "docs", "TEMP_CARS.md")
    with open(cls.mock_file_path, 'w') as f:
        f.write(cls.mock_markdown_content)

    cls.car_info_dict = {
      'Make': 'Acura', 
      'Model': 'ILX 2016-19', 
      'Supported Package': 'AcuraWatch Plus', 
      'ACC': 'openpilot', 
      'No ACC accel below': '25 mph', 
      'No ALC below': '25 mph', 
      'Steering Torque': '[![star](assets/icon-star-empty.svg)](##)', 
      'Resume from stop': '[![star](assets/icon-star-empty.svg)](##)', 
      'Hardware Needed': '<details><summary>Parts</summary><sub>- 1 Honda Nidec connector<br>- 1 RJ45 cable (7 ft)<br>- 1 comma 3X<br>- 1 comma power v2<br>- 1 harness box<br>- 1 mount<br>- 1 right angle OBD-C cable (1.5 ft)<br><a href="https://comma.ai/shop/comma-3x.html?make=Acura&model=ILX 2016-19">Buy Here</a></sub></details>', 
      'Video': '', 
      'Detail sentence': "openpilot upgrades your <strong>Acura ILX</strong> with automated lane centering and adaptive cruise control <strong>while driving above 25 mph</strong>. This car may not be able to take tight turns on its own. Traffic light and stop sign handling is also available in <a href='https://blog.comma.ai/090release/#experimental-mode' target='_blank' class='link-light-new-regular-text'>Experimental mode</a>."
    }

    cls.changes_dict = {'additions': [], 'deletions': [], 'modifications': [], 'detail_sentence_changes': []}

  @classmethod
  def tearDownClass(cls):
    os.remove(cls.mock_file_path)

  def test_column_changes_modification(self):
      old_dict = dict(self.car_info_dict)
      new_dict = dict(old_dict)
      new_dict['Model'] = 'ILX 2016-49'
      modifications_dict = dict(new_dict)
      modifications_dict['modified_fields'] = {'Model': 'ILX 2016-19'}
      expected_changes = dict(self.changes_dict)
      expected_changes['modifications'] = [modifications_dict]

      result = compare_car_info([old_dict], [new_dict])

      self.assertEqual(result, expected_changes, "Modifications not identified correctly")
  
  def test_column_changes_deletion(self):
      old_dict = dict(self.car_info_dict)
      expected_changes = dict(self.changes_dict)
      expected_changes['deletions'] = [old_dict]

      result = compare_car_info([old_dict], [])

      self.assertEqual(result, expected_changes, "Deletion not identified correctly")

  def test_column_changes_addition(self):
      new_dict = dict(self.car_info_dict)
      expected_changes = dict(self.changes_dict)
      expected_changes['additions'] = [new_dict]

      result = compare_car_info([], [new_dict])

      self.assertEqual(result, expected_changes, "Addition not identified correctly")

  def test_process_markdown_file(self):
    car_info = dict(self.car_info_dict)
    expected_data = [car_info]

    processed_data = process_markdown_file(self.mock_file_path)

    self.assertEqual(processed_data, expected_data, "Processed data does not match expected output")


  def test_generator(self):
    generated_cars_md = generate_cars_md(self.all_cars, CARS_MD_TEMPLATE)
    with open(CARS_MD_OUT, "r") as f:
      current_cars_md = f.read()

    self.assertEqual(generated_cars_md, current_cars_md,
                     "Run selfdrive/car/docs.py to update the compatibility documentation")

  def test_docs_diff(self):
    dump_path = os.path.join(BASEDIR, "selfdrive", "car", "tests", "cars_dump")
    dump_car_info(dump_path)
    print_car_info_diff()
    os.remove(dump_path)

  def test_duplicate_years(self):
    make_model_years = defaultdict(list)
    for car in self.all_cars:
      with self.subTest(car_info_name=car.name):
        make_model = (car.make, car.model)
        for year in car.year_list:
          self.assertNotIn(year, make_model_years[make_model], f"{car.name}: Duplicate model year")
          make_model_years[make_model].append(year)

  def test_missing_car_info(self):
    all_car_info_platforms = get_interface_attr("CAR_INFO", combine_brands=True).keys()
    for platform in sorted(interfaces.keys()):
      with self.subTest(platform=platform):
        self.assertTrue(platform in all_car_info_platforms, "Platform: {} doesn't exist in CarInfo".format(platform))

  def test_naming_conventions(self):
    # Asserts market-standard car naming conventions by brand
    for car in self.all_cars:
      with self.subTest(car=car):
        tokens = car.model.lower().split(" ")
        if car.car_name == "hyundai":
          self.assertNotIn("phev", tokens, "Use `Plug-in Hybrid`")
          self.assertNotIn("hev", tokens, "Use `Hybrid`")
          if "plug-in hybrid" in car.model.lower():
            self.assertIn("Plug-in Hybrid", car.model, "Use correct capitalization")
          if car.make != "Kia":
            self.assertNotIn("ev", tokens, "Use `Electric`")
        elif car.car_name == "toyota":
          if "rav4" in tokens:
            self.assertIn("RAV4", car.model, "Use correct capitalization")

  def test_torque_star(self):
    # Asserts brand-specific assumptions around steering torque star
    for car in self.all_cars:
      with self.subTest(car=car):
        # honda sanity check, it's the definition of a no torque star
        if car.car_fingerprint in (HONDA.ACCORD, HONDA.CIVIC, HONDA.CRV, HONDA.ODYSSEY, HONDA.PILOT):
          self.assertEqual(car.row[Column.STEERING_TORQUE], Star.EMPTY, f"{car.name} has full torque star")
        elif car.car_name in ("toyota", "hyundai"):
          self.assertNotEqual(car.row[Column.STEERING_TORQUE], Star.EMPTY, f"{car.name} has no torque star")

  def test_year_format(self):
    for car in self.all_cars:
      with self.subTest(car=car):
        self.assertIsNone(re.search(r"\d{4}-\d{4}", car.name), f"Format years correctly: {car.name}")

  def test_harnesses(self):
    for car in self.all_cars:
      with self.subTest(car=car):
        if car.name == "comma body":
          raise unittest.SkipTest

        car_part_type = [p.part_type for p in car.car_parts.all_parts()]
        car_parts = list(car.car_parts.all_parts())
        self.assertTrue(len(car_parts) > 0, f"Need to specify car parts: {car.name}")
        self.assertTrue(car_part_type.count(PartType.connector) == 1, f"Need to specify one harness connector: {car.name}")
        self.assertTrue(car_part_type.count(PartType.mount) == 1, f"Need to specify one mount: {car.name}")
        self.assertTrue(Cable.right_angle_obd_c_cable_1_5ft in car_parts, f"Need to specify a right angle OBD-C cable (1.5ft): {car.name}")

if __name__ == "__main__":
  unittest.main()
