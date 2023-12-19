import jinja2

from cereal import car
Ecu = car.CarParams.Ecu

from openpilot.selfdrive.car.interfaces import get_interface_attr

CARS = get_interface_attr('CAR')
FW_VERSIONS = get_interface_attr('FW_VERSIONS')
FINGERPRINTS = get_interface_attr('FINGERPRINTS')
PLATFORM_TO_PYTHON_CAR_NAME = {brand: {car.value: car.name for car in CARS[brand]} for brand in CARS}
ECU_NUMBER_TO_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}

FINGERPRINTS_PY_TEMPLATE = """{% if FINGERPRINTS[brand] %}
# ruff: noqa: E501
{% endif %}
{% if FW_VERSIONS[brand] %}
from cereal import car
{% endif %}
from openpilot.selfdrive.car.{{brand}}.values import CAR
{% if FW_VERSIONS[brand] %}
Ecu = car.CarParams.Ecu
{% endif %}

{% if FINGERPRINTS[brand] %}
FINGERPRINTS = {
{% for car, fingerprints in FINGERPRINTS[brand].items() %}
  CAR.{{PLATFORM_TO_PYTHON_CAR_NAME[brand][car]}}: [
  {% for fingerprint in fingerprints %}    { {% for key, value in fingerprint.items() %}{{key}}: {{value}}, {% endfor %}},
  {% endfor %}
  ],
{% endfor %}
}
{% endif %}

{% if FW_VERSIONS[brand] %}
FW_VERSIONS = {
{% for car, _ in FW_VERSIONS[brand].items() %}
  CAR.{{PLATFORM_TO_PYTHON_CAR_NAME[brand][car]}}: {
  {% for key, fw_versions in FW_VERSIONS[brand][car].items() %}
    (Ecu.{{ECU_NUMBER_TO_NAME[key[0]]}}, 0x{{"%0x" | format(key[1] | int)}}, \
{% if key[2] %}0x{{"%0x" | format(key[2] | int)}}{% else %}{{key[2]}}{% endif %}): [
    {% for fw_version in fw_versions %}
      {{fw_version}},
    {% endfor %}
    ],
  {% endfor %}
  },
{% endfor %}
}
{% endif %}
"""

def format_brand_fw_versions(brand):
  with open(f"selfdrive/car/{brand}/fingerprints.py", "w") as f:
    template = jinja2.Template(FINGERPRINTS_PY_TEMPLATE, trim_blocks=True, lstrip_blocks=True)

    f.write(template.render(brand=brand, ECU_NUMBER_TO_NAME=ECU_NUMBER_TO_NAME, PLATFORM_TO_PYTHON_CAR_NAME=PLATFORM_TO_PYTHON_CAR_NAME,
                            FINGERPRINTS=FINGERPRINTS, FW_VERSIONS=FW_VERSIONS))

for brand in FW_VERSIONS.keys():
  format_brand_fw_versions(brand)
