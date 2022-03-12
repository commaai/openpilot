# Supported Cars

Cars are organized into three tiers:

- 🥇 Gold - a high-quality openpilot experience
- 🥈 Silver - a pretty good, albeit limited experience
- 🥉 Bronze - a significantly limited experience

|                          Star                          | openpilot Longitudinal                                                 | Star                                                            | Full-Speed Range (FSR) Longitudinal                   |
|:---------------------------------------------------:|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| <img src="assets/icon-star-full.png" width="22" />  | openpilot is able to control gas and brakes                            | <img src="assets/icon-star-full.png" width="22" /> |  Adaptive Cruise Control (ACC) operates down to 0 mph |
| <img src="assets/icon-star-half.png" width="22" />  | openpilot is able to control the gas and brakes with some restrictions |<img src="assets/icon-star-half.png" width="22" />
| <img src="assets/icon-star-empty.png" width="22" /> | The gas and brakes are controlled by the car's stock Adaptive Cruise Control (ACC) system |

|                                                                                                                                                                                openpilot Longitudinal                                                                                                                                                                                | Full-Speed Range (FSR) Longitudinal                                                 |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|------------------------------------------------------------------------|
| <img src="assets/icon-star-full.png" width="22" /> - openpilot is able to control gas and brakes<br/><img src="assets/icon-star-half.png" width="22" /> - openpilot is able to control the gas and brakes with some restrictions<br/><img src="assets/icon-star-empty.png" width="22" /> - The gas and brakes are controlled by the car's stock Adaptive Cruise Control (ACC) system |

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
