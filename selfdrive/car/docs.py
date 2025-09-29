#!/usr/bin/env python3
import os

from openpilot.common.basedir import BASEDIR
from opendbc.car.docs import get_all_car_docs, generate_cars_md

CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "selfdrive", "car", "CARS_template.md")


if __name__ == "__main__":
  doc_path = get_doc_path()
  if doc_path is None:
    print("No doc path found, run from a car brand directory", file=sys.stderr)
    sys.exit(1)

  car_docs = get_car_docs()

  with open(doc_path, "w") as f:
    json.dump(car_docs, f, indent=2, sort_keys=True)
