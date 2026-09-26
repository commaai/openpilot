#!/usr/bin/env python3
import argparse
import os
from string import Template

from openpilot.common.basedir import BASEDIR
from opendbc.car.docs import get_all_car_docs, get_all_footnotes
from opendbc.car.docs_definitions import Column, SupportType

CARS_MD_OUT = os.path.join(BASEDIR, "docs", "CARS.md")
CARS_MD_TEMPLATE = os.path.join(BASEDIR, "openpilot/selfdrive", "car", "CARS_template.md")

FOOTNOTE_TAG = '[<sup>{}</sup>](#footnotes)'
STAR_ICON = '[![star](assets/icon-star-{}.svg)](##)'
VIDEO_ICON = '<a href="{}" target="_blank"><img height="18px" src="assets/icon-youtube.svg" /></a>'
# Force hardware column wider by using a blank image with max width.
HARDWARE_COL_NAME = 'Hardware Needed'
WIDE_HARDWARE_COL_NAME = f'<a href="##"><img width=2000></a>{HARDWARE_COL_NAME}<br>&nbsp;'


def _build_cars_table(upstream_cars) -> tuple[str, str, str]:
  columns = list(Column)
  header_cells = [
    WIDE_HARDWARE_COL_NAME if col.value == HARDWARE_COL_NAME else col.value
    for col in columns
  ]
  table_header = "|" + "|".join(header_cells) + "|"

  # First three columns left-aligned (---), remaining centered (:---:)
  sep_parts = ["---"] * min(3, len(columns)) + [":---:"] * max(0, len(columns) - 3)
  table_separator = "|" + "|".join(sep_parts) + "|"

  rows = []
  for car_docs in upstream_cars:
    cells = [car_docs.get_column(column, STAR_ICON, VIDEO_ICON, FOOTNOTE_TAG) for column in columns]
    rows.append("|" + "|".join(cells) + "|")
  table_rows = "\n".join(rows) + ("\n" if rows else "")

  return table_header, table_separator, table_rows


def generate_cars_md(all_car_docs, template_fn: str, **kwargs) -> str:
  del kwargs  # kept for call-site compatibility

  upstream_cars = [c for c in all_car_docs if c.support_type == SupportType.UPSTREAM]
  table_header, table_separator, table_rows = _build_cars_table(upstream_cars)

  footnotes = [fn.value.text.replace('</br>', '') for fn in get_all_footnotes()]
  footnotes_md = "\n".join(
    f"<sup>{i}</sup>{text} <br />"
    for i, text in enumerate(footnotes, start=1)
  ) + ("\n" if footnotes else "")

  with open(template_fn) as f:
    template = Template(f.read())

  return template.substitute(
    supported_count=len(upstream_cars),
    table_header=table_header,
    table_separator=table_separator,
    table_rows=table_rows,
    footnotes=footnotes_md,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto generates supported cars documentation",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--template", default=CARS_MD_TEMPLATE, help="Override default template filename")
  parser.add_argument("--out", default=CARS_MD_OUT, help="Override default generated filename")
  args = parser.parse_args()

  with open(args.out, 'w') as f:
    f.write(generate_cars_md(get_all_car_docs(), args.template))
  print(f"Generated and written to {args.out}")
