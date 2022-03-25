{% set footnote_tag = '[<sup>{}</sup>](#Footnotes)' -%}
{% set star_icon = '<a href="#"><img valign="top" src="assets/icon-star-{}.svg" width="22" /></a>' -%}

# Supported Cars

A supported vehicle is one that just works when you install a comma device. Every car performs differently with openpilot, but all supported cars should provide a better experience than any stock system.

Cars are organized into three tiers:

{% for tier in tiers %}
- {{tier.name.title()}} - {{tier.value}}
{% endfor %}

How We Rate The Cars
---

### openpilot Adaptive Cruise Control (ACC)
- {{star_icon.format(Star.FULL.value)}} - openpilot is able to control the gas and brakes.
- {{star_icon.format(Star.HALF.value)}} - openpilot is able to control the gas and brakes with some restrictions.
- {{star_icon.format(Star.EMPTY.value)}} - The gas and brakes are controlled by the car's stock Adaptive Cruise Control (ACC) system.

### Stop and Go
- {{star_icon.format(Star.FULL.value)}} - Adaptive Cruise Control (ACC) operates down to 0 mph.
- {{star_icon.format(Star.EMPTY.value)}} - Adaptive Cruise Control (ACC) available only above certain speeds. See your car's manual for the minimum speed.

### Steer to 0
- {{star_icon.format(Star.FULL.value)}} - openpilot can control the steering wheel down to 0 mph.
- {{star_icon.format(Star.EMPTY.value)}} - No steering control below certain speeds.

### Steering Torque
- {{star_icon.format(Star.FULL.value)}} - Car has enough steering torque for comfortable highway driving.
- {{star_icon.format(Star.EMPTY.value)}} - Limited ability to make turns.

### Actively Maintained
- {{star_icon.format(Star.FULL.value)}} - Mainline software support, harness hardware sold by comma, lots of users, primary development target.
- {{star_icon.format(Star.EMPTY.value)}} - Low user count, community maintained, harness hardware not sold by comma.

**All supported cars can move between the tiers as support changes.**

{% for tier, cars in tiers.items() %}
## {{tier.name.title()}} Cars

|{{Column | map(attribute='value') | join('|')}}|
|---|---|---|:---:|:---:|:---:|:---:|:---:|
{% for car_info in cars %}
|{% for column in Column %}{{car_info.get_column(column, star_icon, footnote_tag)}}|{% endfor %}

{% endfor %}

{% endfor %}

## Footnotes
{% for footnote in footnotes %}
<sup>{{loop.index}}</sup>{{footnote}} <br />
{% endfor %}

## Community Maintained Cars
Although they're not upstream, the community has openpilot running on other makes and models. See the 'Community Supported Models' section of each make [on our wiki](https://wiki.comma.ai/).
