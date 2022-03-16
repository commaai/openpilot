# Supported Cars

A supported vehicle is one that just works when you install openpilot on a compatible device. Every car performs differently with openpilot, but we aim for all supported cars to provide a solid highway experience in the US market.

Cars are organized into three tiers:

- Gold - The best openpilot experience. Great highway driving with continual updates.
- Silver - A solid highway experience, but is limited by stock longitudinal.
- Bronze - A solid highway experience, but will have limited performance in stop-and-go. May have ACC and ALC speed limitations.

How We Rate The Cars
---

### openpilot Adaptive Cruise Control (ACC)
- {{star_full}} - openpilot is able to control gas and brakes
- {{star_half}} - openpilot is able to control the gas and brakes with some restrictions
- {{star_empty}} - The gas and brakes are controlled by the car's stock Adaptive Cruise Control (ACC) system

### Stop and Go
- {{star_full}} - Adaptive Cruise Control (ACC) operates down to 0 mph
- {{star_empty}} - Adaptive Cruise Control (ACC) available only above certain speeds. See your car's manual for the minimum speed.

### Steer to 0
- {{star_full}} - openpilot can control the steering wheel down to 0 mph
- {{star_empty}} - No steering control below certain speeds

### Steering Torque
- {{star_full}} - Car has enough steering torque for comfortable highway driving
- {{star_empty}} - Limited ability to make turns

### Actively Maintained
- {{star_full}} - Mainline software support, harness hardware sold by comma, lots of users, primary development target
- {{star_empty}} - Low user count, community maintained, harness hardware not sold by comma

**All supported cars can move between the tiers as support changes.**

{% for tier, car_rows in tiers %}
## {{tier}} Cars

|{{columns | join('|')}}|
|---|---|---|:---:|:---:|:---:|:---:|:---:|
{% for row in car_rows %}
|{{row | join('|')}}|
{% endfor %}

{% endfor %}

## Footnotes

{% for footnote in footnotes %}
<sup>{{loop.index}}</sup>{{footnote}} <br />
{% endfor %}

## Community Maintained Cars
Although they're not upstream, the community has openpilot running on other makes and models. See the 'Community Supported Models' section of each make [on our wiki](https://wiki.comma.ai/).
