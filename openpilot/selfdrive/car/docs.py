#!/usr/bin/env python3
import argparse
import os

import jinja2

from openpilot.common.basedir import BASEDIR
from opendbc.car.docs import get_all_car_docs, get_all_footnotes, group_by_make
from opendbc.car.docs_definitions import BaseCarHarness, Column, Device, ExtraCarsColumn, PartType, SupportType

CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "openpilot/selfdrive", "car", "CARS_template.md")


def generate_cars_md(all_car_docs, template_fn: str, **kwargs) -> str:
  with open(template_fn) as f:
    template = jinja2.Template(f.read(), trim_blocks=True, lstrip_blocks=True)

  footnotes = [fn.value.text for fn in get_all_footnotes()]
  return template.render(all_car_docs=all_car_docs, PartType=PartType,
                         group_by_make=group_by_make, footnotes=footnotes,
                         Device=Device, Column=Column, ExtraCarsColumn=ExtraCarsColumn,
                         BaseCarHarness=BaseCarHarness, SupportType=SupportType,
                         **kwargs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supported cars documentation",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--template", default=CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_all_car_docs(), args.template))
  print(f"Generated and written to {args.out}")
