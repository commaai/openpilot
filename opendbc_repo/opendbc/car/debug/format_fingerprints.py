#!/usr/bin/env python3
import jinja2
import os

from opendbc.car.common.basedir import BASEDIR
from opendbc.car.interfaces import get_interface_attr
from opendbc.car.structs import CarParams

Ecu = CarParams.Ecu

CARS = get_interface_attr('CAR')
FW_VERSIONS = get_interface_attr('FW_VERSIONS')
FINGERPRINTS = get_interface_attr('FINGERPRINTS')
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}

FINGERPRINTS_PY_TEMPLATE = jinja2.Template("""
{%- if FINGERPRINTS[brand] and brand != 'body' %}
# ruff: noqa: E501
{% endif %}
\"\"\" AUTO-FORMATTED USING opendbc/car/debug/format_fingerprints.py, EDIT STRUCTURE THERE.\"\"\"
{% if FW_VERSIONS[brand] %}
from opendbc.car.structs import CarParams
{% endif %}
from opendbc.car.{{brand}}.values import CAR
{% if FW_VERSIONS[brand] %}

Ecu = CarParams.Ecu
{% endif %}
{% if comments +%}
{{ comments | join() }}
{% endif %}
{% if FINGERPRINTS[brand] %}

FINGERPRINTS = {
{% for car, fingerprints in FINGERPRINTS[brand].items() %}
  CAR.{{car.name}}: [{
{% for fingerprint in fingerprints %}
{% if not loop.first %}
  {{ "{" }}
{% endif %}
    {% for key, value in fingerprint.items() %}{{key}}: {{value}}{% if not loop.last %}, {% endif %}{% endfor %}

  }{% if loop.last %}]{% endif %},
{% endfor %}
{% endfor %}
}
{% endif %}

FW_VERSIONS{% if not FW_VERSIONS[brand] %}: dict[str, dict[tuple, list[bytes]]]{% endif %} = {
{% for car, _ in FW_VERSIONS[brand].items() %}
  CAR.{{car.name}}: {
{% for key, fw_versions in FW_VERSIONS[brand][car].items() %}
    (Ecu.{{ECU_NAME[key[0]]}}, 0x{{"%0x" | format(key[1] | int)}}, \
{% if key[2] %}0x{{"%0x" | format(key[2] | int)}}{% else %}{{key[2]}}{% endif %}): [
  {% for fw_version in (fw_versions + extra_fw_versions.get(car, {}).get(key, [])) | unique | sort %}
    {{fw_version}},
  {% endfor %}
  ],
{% endfor %}
  },
{% endfor %}
}

""", trim_blocks=True)


def format_brand_fw_versions(brand, extra_fw_versions: None | dict[str, dict[tuple, list[bytes]]] = None):
  extra_fw_versions = extra_fw_versions or {}

  fingerprints_file = os.path.join(BASEDIR, f"{brand}/fingerprints.py")
  with open(fingerprints_file) as f:
    comments = [line for line in f.readlines() if line.startswith("#") and "noqa" not in line]

  with open(fingerprints_file, "w") as f:
    f.write(FINGERPRINTS_PY_TEMPLATE.render(brand=brand, comments=comments, ECU_NAME=ECU_NAME,
                                            FINGERPRINTS=FINGERPRINTS, FW_VERSIONS=FW_VERSIONS,
                                            extra_fw_versions=extra_fw_versions))


if __name__ == "__main__":
  for brand in FW_VERSIONS.keys():
    format_brand_fw_versions(brand)
