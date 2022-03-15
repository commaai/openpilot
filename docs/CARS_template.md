# Supported Cars

A supported vehicle is one that just works when you install openpilot on a compatible device. Every car performs differently with openpilot, but we aim for all supported cars to provide a solid highway experience in the US market.

Cars are organized into three tiers:

- Gold - The best openpilot experience. Great highway driving with continual updates.
- Silver - A solid highway experience, but is limited by stock longitudinal.
- Bronze - A solid highway experience, but will have limited performance in stop-and-go. May have ACC and ALC speed limitations.

How We Rate The Cars
---
<table>
  <tr>
    <th width="50%">openpilot Longitudinal</th>
    <th>Full-Speed Range (FSR) Longitudinal</th>
  </tr>
    <td valign="top"><img style="float: left;" src="assets/icon-star-full.svg" width="22" /> - openpilot is able to control gas and brakes<br/><img style="float: left;" src="assets/icon-star-half.svg" width="22" /> - openpilot is able to control the gas and brakes with some restrictions<br/><img style="float: left;" src="assets/icon-star-empty.svg" width="22" /> - The gas and brakes are controlled by the car's stock Adaptive Cruise Control (ACC) system</td>
    <td valign="top"><img style="float: left;" src="assets/icon-star-full.svg" width="22" /> - Adaptive Cruise Control (ACC) operates down to 0 mph<br/><img style="float: left;" src="assets/icon-star-empty.svg" width="22" /> - Adaptive Cruise Control (ACC) available only above certain speeds. See your car's manual for the minimum speed.</td>
  </tr>
</table>

<table>
  <tr>
    <th width="50%">Full-Speed Range (FSR) Steering</th>
    <th>Steering Torque</th>
  </tr>
    <td valign="top"><img style="float: left;" src="assets/icon-star-full.svg" width="22" /> - openpilot can control the steering wheel down to 0 mph<br/><img style="float: left;" src="assets/icon-star-empty.svg" width="22" /> - No steering control below certain speeds</td>
    <td valign="top"><img style="float: left;" src="assets/icon-star-full.svg" width="22" /> - Car has enough steering torque for comfortable highway driving<br/><img style="float: left;" src="assets/icon-star-empty.svg" width="22" /> - Limited ability to make turns</td>
  </tr>
</table>

<table >
  <tr>
    <th>Actively Maintained</th>
    <td style="visibility:hidden;" width="50%"></td>
  </tr>
    <td valign="top"><img style="float: left;" src="assets/icon-star-full.svg" width="22" /> - Mainline software support, harness hardware sold by comma, lots of users, primary development target<br/><img style="float: left;" src="assets/icon-star-empty.svg" width="22" /> - Low user count, community maintained, harness hardware sold by comma</td>
  </tr>
</table>

**All supported cars can move between the tiers as support changes.**

{% for tier, car_rows in tiers %}
## {{tier}} Cars

|{{columns | join('|')}}|
|---|---|---|:---:|:---:|:---:|:---:|:---:|
{% for row in car_rows %}
|{{row | join('|')}}|
{% endfor %}

{% endfor %}

{% for footnote in footnotes %}
<sup>{{loop.index}}</sup>{{footnote}} <br />
{% endfor %}

## Community Maintained Cars
Although they're not upstream, the community has openpilot running on other makes and models. See the 'Community Supported Models' section of each make [on our wiki](https://wiki.comma.ai/).
