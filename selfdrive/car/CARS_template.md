# Supported Cars

A supported vehicle is one that just works when you install a comma device. Every car performs differently with openpilot, but all supported cars should provide a better experience than any stock system.

Cars are organized into three tiers:

{% for tier, car_rows in tiers %}
- {{tier.name.title()}} - {{tier.value}}
{% endfor %}

How We Rate The Cars
---

### openpilot Adaptive Cruise Control (ACC)
- {{Star.FULL.md_icon}} - openpilot is able to control the gas and brakes.
- {{Star.HALF.md_icon}} - openpilot is able to control the gas and brakes with some restrictions.
- {{Star.EMPTY.md_icon}} - The gas and brakes are controlled by the car's stock Adaptive Cruise Control (ACC) system.

### Stop and Go
- {{Star.FULL.md_icon}} - Adaptive Cruise Control (ACC) operates down to 0 mph.
- {{Star.EMPTY.md_icon}} - Adaptive Cruise Control (ACC) available only above certain speeds. See your car's manual for the minimum speed.

### Steer to 0
- {{Star.FULL.md_icon}} - openpilot can control the steering wheel down to 0 mph.
- {{Star.EMPTY.md_icon}} - No steering control below certain speeds.

### Steering Torque
- {{Star.FULL.md_icon}} - Car has enough steering torque for comfortable highway driving.
- {{Star.EMPTY.md_icon}} - Limited ability to make turns.

### Actively Maintained
- {{Star.FULL.md_icon}} - Mainline software support, harness hardware sold by comma, lots of users, primary development target.
- {{Star.EMPTY.md_icon}} - Low user count, community maintained, harness hardware not sold by comma.

**All supported cars can move between the tiers as support changes.**

{% set footnote_tag = '[<sup>{}</sup>](#Footnotes)' %}
{% for tier, car_rows in tiers %}
## {{tier.name.title()}} Cars

|{{columns | join('|')}}|
|---|---|---|:---:|:---:|:---:|:---:|:---:|
{% for row in car_rows %}
|{% for row_item in row %}{{row_item.text if row_item.text else row_item.star.md_icon}}{{footnote_tag.format(row_item.footnote) if row_item.footnote else ''}}|{% endfor %}

{% endfor %}

{% endfor %}

{% for footnote in footnotes %}
<sup>{{loop.index}}</sup>{{footnote}} <br />
{% endfor %}

## Community Maintained Cars
Although they're not upstream, the community has openpilot running on other makes and models. See the 'Community Supported Models' section of each make [on our wiki](https://wiki.comma.ai/).
