#!/usr/bin/env python3
import re
import os
import argparse
import unicodedata
from string import Template
from typing import get_args

from enum import Enum

from opendbc.car.common.basedir import BASEDIR
from opendbc.car import gen_empty_fingerprint
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import CarDocs, ExtraCarDocs, ExtraCarsColumn, CommonFootnote
from opendbc.car.car_helpers import interfaces
from opendbc.car.interfaces import get_interface_attr
from opendbc.car.values import Platform
from opendbc.car.mock.values import CAR as MOCK
from opendbc.car.extra_cars import CAR as EXTRA


EXTRA_CARS_MD_OUT = os.path.join(BASEDIR, "../", "../", "docs", "CARS.md")
EXTRA_CARS_MD_TEMPLATE = os.path.join(BASEDIR, "CARS_template.md")

# TODO: merge these platforms into normal car ports with SupportType flag
ExtraPlatform = Platform | EXTRA
EXTRA_BRANDS = get_args(ExtraPlatform)
EXTRA_PLATFORMS: dict[str, ExtraPlatform] = {str(platform): platform for brand in EXTRA_BRANDS for platform in brand}


def get_params_for_docs(platform) -> CarParams:
  cp_platform = platform if platform in interfaces else MOCK.MOCK
  CP: CarParams = interfaces[cp_platform].get_params(cp_platform, fingerprint=gen_empty_fingerprint(),
                                                     car_fw=[CarParams.CarFw(ecu=CarParams.Ecu.unknown)],
                                                     alpha_long=True, is_release=True, docs=True)
  return CP


def get_all_footnotes() -> dict[Enum, int]:
  all_footnotes = list(CommonFootnote)
  for footnotes in get_interface_attr("Footnote", ignore_none=True).values():
    all_footnotes.extend(footnotes)
  return {fn: idx + 1 for idx, fn in enumerate(all_footnotes)}


def _natural_sort_key(s):
  # NFKD normalization ensures accented characters sort with their base letter (e.g., Š sorts with S)
  normalized = unicodedata.normalize('NFKD', s)
  return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', normalized) if t]


def build_sorted_car_docs_list(platforms, footnotes=None):
  collected_car_docs: list[CarDocs | ExtraCarDocs] = []
  for platform in platforms.values():
    car_docs = platform.config.car_docs
    CP = get_params_for_docs(platform)

    if not len(car_docs):
      continue

    # A platform can include multiple car models
    for _car_docs in car_docs:
      if not hasattr(_car_docs, "row"):
        _car_docs.init_make(CP)
        _car_docs.init(CP, footnotes)
      collected_car_docs.append(_car_docs)

  # Sort cars by make and model + year
  sorted_cars = sorted(collected_car_docs, key=lambda car: _natural_sort_key(car.name))
  return sorted_cars


# CAUTION: This function is imported by shop.comma.ai and comma.ai/vehicles, test changes carefully
def get_all_car_docs() -> list[CarDocs]:
  collected_footnotes = get_all_footnotes()
  sorted_list: list[CarDocs] = build_sorted_car_docs_list(EXTRA_PLATFORMS, footnotes=collected_footnotes)
  return sorted_list


def _build_cars_table(all_car_docs: list[CarDocs], **kwargs) -> tuple[str, str, str]:
  """Build markdown table header, separator, and body rows for ExtraCarsColumn."""
  hardware_col_name = kwargs.get("hardware_col_name", "")
  wide_hardware_col_name = kwargs.get("wide_hardware_col_name", "")

  columns = list(ExtraCarsColumn)
  header_cells = [col.value for col in columns]
  if hardware_col_name:
    header_cells = [wide_hardware_col_name if c == hardware_col_name else c for c in header_cells]
  table_header = "|" + "|".join(header_cells) + "|"

  # First three columns left-aligned (---), remaining centered (:---:)
  sep_parts = ["---"] * min(3, len(columns)) + [":---:"] * max(0, len(columns) - 3)
  table_separator = "|" + "|".join(sep_parts) + "|"

  rows = []
  for car_docs in all_car_docs:
    cells = [car_docs.get_extra_cars_column(column) for column in columns]
    rows.append("|" + "|".join(cells) + "|")
  table_rows = "\n".join(rows) + ("\n" if rows else "")

  return table_header, table_separator, table_rows


# CAUTION: This function is imported by shop.comma.ai and comma.ai/vehicles, test changes carefully
def generate_cars_md(all_car_docs: list[CarDocs], template_fn: str, **kwargs) -> str:
  with open(template_fn) as f:
    template = Template(f.read())

  table_header, table_separator, table_rows = _build_cars_table(all_car_docs, **kwargs)
  cars_md: str = template.substitute(
    car_count=len(all_car_docs),
    table_header=table_header,
    table_separator=table_separator,
    table_rows=table_rows,
  )
  # Match historical output: no trailing newline at EOF
  return cars_md.rstrip("\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supportability info docs for all known cars",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--template", default=EXTRA_CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=EXTRA_CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_all_car_docs(), args.template))
  print(f"Generated and written to {args.out}")
