# Supported Cars

Cars are organized into three tiers:

- 🥇 Gold - a high-quality openpilot experience
- 🥈 Silver - a pretty good, albeit limited experience
- 🥉 Bronze - a significantly limited experience

Tier Criteria:

- openpilot Longitudinal - openpilot is able to control gas and brakes. If no star is present, the car is limited to the stock system
- FSR Longitudinal - openpilot can brake and accelerate down to 0 mph
- FSR Steering - openpilot can actuate the steering wheel down to 0 mph
- Steering Torque - car has enough steering torque for comfortable highway driving
- Actively Maintained - mainline software support, harness hardware sold by comma.ai

**All supported cars can move between the tiers as support changes.**

{% for tier, car_rows in tiers %}
## {{tier}} Cars

|{{columns | join('|')}}|
|---|---|---|:---:|:---:|:---:|:---:|:---:|
{% for row in car_rows %}
|{{row | join('|')}}|
{% endfor %}

{% endfor %}

{% for exception in exceptions %}
<sup>{{loop.index}}</sup>{{exception}} <br />
{% endfor %}

## Community Maintained Cars
Although they're not upstream, the community has openpilot running on other makes and models. See the 'Community Supported Models' section of each make [on our wiki](https://wiki.comma.ai/).
