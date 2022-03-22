# Supported Cars

A supported vehicle is one that just works when you install a comma device. Every car performs differently with openpilot, but all supported cars should provide a better experience than any stock system.

Cars are organized into three tiers:

- Gold - The best openpilot experience. Great highway driving and beyond.
- Silver - A solid highway driving experience, but is limited by stock longitudinal. May be upgraded in the future.
- Bronze - A good highway experience, but may have limited performance in traffic and on sharp turns.

How We Rate The Cars
---

### openpilot Adaptive Cruise Control (ACC)
- <a href="#"><img valign="top" src="assets/icon-star-full.svg" width="22" /></a> - openpilot is able to control the gas and brakes.
- <a href="#"><img valign="top" src="assets/icon-star-half.svg" width="22" /></a> - openpilot is able to control the gas and brakes with some restrictions.
- <a href="#"><img valign="top" src="assets/icon-star-empty.svg" width="22" /></a> - The gas and brakes are controlled by the car's stock Adaptive Cruise Control (ACC) system.

### Stop and Go
- <a href="#"><img valign="top" src="assets/icon-star-full.svg" width="22" /></a> - Adaptive Cruise Control (ACC) operates down to 0 mph.
- <a href="#"><img valign="top" src="assets/icon-star-empty.svg" width="22" /></a> - Adaptive Cruise Control (ACC) available only above certain speeds. See your car's manual for the minimum speed.

### Steer to 0
- <a href="#"><img valign="top" src="assets/icon-star-full.svg" width="22" /></a> - openpilot can control the steering wheel down to 0 mph.
- <a href="#"><img valign="top" src="assets/icon-star-empty.svg" width="22" /></a> - No steering control below certain speeds.

### Steering Torque
- <a href="#"><img valign="top" src="assets/icon-star-full.svg" width="22" /></a> - Car has enough steering torque for comfortable highway driving.
- <a href="#"><img valign="top" src="assets/icon-star-empty.svg" width="22" /></a> - Limited ability to make turns.

### Actively Maintained
- <a href="#"><img valign="top" src="assets/icon-star-full.svg" width="22" /></a> - Mainline software support, harness hardware sold by comma, lots of users, primary development target.
- <a href="#"><img valign="top" src="assets/icon-star-empty.svg" width="22" /></a> - Low user count, community maintained, harness hardware not sold by comma.

**All supported cars can move between the tiers as support changes.**


<sup>1</sup>2019 Honda Civic 1.6L Diesel Sedan does not have ALC below 12mph. <br />
<sup>2</sup>Requires an [OBD-II](https://comma.ai/shop/products/comma-car-harness) car harness and [community built ASCM harness](https://github.com/commaai/openpilot/wiki/GM#hardware). NOTE: disconnecting the ASCM disables Automatic Emergency Braking (AEB). <br />
<sup>3</sup>Not including the China market Kamiq, which is based on the (currently) unsupported PQ34 platform. <br />
<sup>4</sup>Not including the USA/China market Passat, which is based on the (currently) unsupported PQ35/NMS platform. <br />
<sup>5</sup>Model-years 2021 and beyond may have a new camera harness design, which isn't yet available from the comma store. Before ordering, remove the Lane Assist camera cover and check to see if the connector is black (older design) or light brown (newer design). For the newer design, in the interim, choose "VW J533 Development" from the vehicle drop-down for a harness that integrates at the CAN gateway inside the dashboard. <br />
<sup>6</sup>When disconnecting the Driver Support Unit (DSU), openpilot Adaptive Cruise Control (ACC) will replace stock Adaptive Cruise Control (ACC). NOTE: disconnecting the DSU disables Automatic Emergency Braking (AEB). <br />
<sup>7</sup>28mph for Camry 4CYL L, 4CYL LE and 4CYL SE which don't have Full-Speed Range Dynamic Radar Cruise Control. <br />
<sup>8</sup>An inaccurate steering wheel angle sensor makes precise control difficult. <br />

## Community Maintained Cars
Although they're not upstream, the community has openpilot running on other makes and models. See the 'Community Supported Models' section of each make [on our wiki](https://wiki.comma.ai/).