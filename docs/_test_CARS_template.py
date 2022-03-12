import os
import jinja2

from common.basedir import BASEDIR

with open(os.path.join(BASEDIR, "docs", "CARS_template.md"), 'r') as f:
  template = f.read()

t = jinja2.Template(template, trim_blocks=True, lstrip_blocks=True)

rendered = t.render(tiers=[('Gold', ["FSR Long", "Other Lat"], []), ('Silver', ["FSR Long"], [])])
print(rendered)
