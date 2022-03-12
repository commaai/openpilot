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

{% for tier, categories, cars in tiers %}
## {{tier}} Cars

|{{categories | join('|')}}|
|---|---|---|:---:|:---:|:---:|:---:|:---:|
{% for car in cars %}
|{{car | join('|')}}|
{% endfor %}

{% endfor %}

<sup>1</sup>When disconnecting the Driver Support Unit (DSU), openpilot ACC will replace stock ACC. ***NOTE: disconnecting the DSU disables Automatic Emergency Braking (AEB).*** <br />
<sup>2</sup>28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control. <br />
<sup>3</sup>When disabling the radar, openpilot ACC will replace stock ACC. ***NOTE: disabling the radar disables Automatic Emergency Braking (AEB).*** <br />
<sup>4</sup>Requires an [OBD-II car harness](https://comma.ai/shop/products/comma-car-harness) and [community built ASCM harness](https://github.com/commaai/openpilot/wiki/GM#hardware). ***NOTE: disconnecting the ASCM disables Automatic Emergency Braking (AEB).*** <br />
<sup>5</sup>Not including the China market Kamiq, which is based on the (currently) unsupported PQ34 platform. <br />
<sup>6</sup>Not including the USA/China market Passat, which is based on the (currently) unsupported PQ35/NMS platform. <br />
<sup>7</sup>Model-years 2021 and beyond may have a new camera harness design, which isn't yet available from the comma store. Before ordering,
remove the Lane Assist camera cover and check to see if the connector is black (older design) or light brown (newer design). For the newer design,
in the interim, choose "VW J533 Development" from the vehicle drop-down for a harness that integrates at the CAN gateway inside the dashboard.<br />
Community Maintained Cars and Features are not verified by comma to meet our [safety model](SAFETY.md). Be extra cautious using them.

## Community Maintained Cars
Although they're not upstream, the community has openpilot running on other makes and models. See the 'Community Supported Models' section of each make [on our wiki](https://wiki.comma.ai/).
