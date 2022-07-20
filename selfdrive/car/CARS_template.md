{% set footnote_tag = '[<sup>{}</sup>](#footnotes)' -%}
{% set star_icon = '<a href="##"><img valign="top" src="assets/icon-star-{}.svg" width="22" /></a>' -%}

# Supported Cars

A supported vehicle is one that just works when you install a comma device. Every car performs differently with openpilot, but all supported cars should provide a better experience than any stock system.

# Don't see your car here?

openpilot *can* support many more cars than it currently does.
There are a few reasons your car may not be supported.
If your car doesn't fit into any of the incompatibility criteria here, then there's a good chance it can be supported! We're adding support for new cars all the time.

### too old

openpilot uses the existing steering, gas, and brake interfaces in your car. If your car lacks any one of these interfaces, openpilot will very . If your car has a form of LKAS/LCA and ACC, then it can almost has these interfaces.

This generaly means openpilot only works on cars made after 2015, though some Volkswagen Group cars are a notable excception to this.
openpilot will never support your car if it does not have an EPS.

### Notes about specific brands

* Toyota/Lexus: Any car that has Toyota/Lexus Safety Sense with "Lane Departure Alert with Steering Assist (LDA w/SA)" and/or "Lane Tracing Assist (LTA)" are candidates. Note that LDA without Steering Assist will not work. This is standard on most newer models.
* Subaru: Any car with EyeSight should be able to be supported. EyeSight is standard on many newer models.
* Honda/Acura: ?
* Hyundai/Kia/Genesis: Any car with Smart Cruise Control (SCC) and Lane Following Assist (LFA) or Lane Keeping Assist (LKAS) will work. LKAS/LFA are standard on most newer cars. SCC may be referred to as NSCC, but it's the same thing.
* Chrysler/Jeep/Ram?
* Volkswagen?
* Nissan: Any car with ProPILOT will likely be able to supported

### FlexRay

All the cars that openpilot supports use a [CAN bus](https://en.wikipedia.org/wiki/CAN_bus) to communicate between all the ECUs, however a CAN bus isn't the only way that the cars in your computer can communicate.

Most, if not all, vehicles from the following manufacturers use [FlexRay](https://en.wikipedia.org/wiki/FlexRay) instead of a CAN bus: BMW, Mercedes, Audi, Land Rover, and some Volvo. These cars may one day be supported, but we have no immediate plans to support FlexRay.

### Toyota Security

So far, this list includes:
* Toyota Rav4 Prime
* Toyota Sienna 2021+
* Toyota Venza 2021+
* Toyota Sequoia 2023+
* Toyota Tundra 2022+
* Toyota Yaris 2022+
* Toyota Corolla Cross (only US model)
* Lexus NX 2021+

### Misc models that aren't supported

TODO: flesh this out more. different page?

* Tesla Model S - different steering
* a bunch of mazda's - steering lockout

How We Rate The Cars
---

{% for star_row in star_descriptions.values() %}
{% for name, stars in star_row.items() %}
### {{name}}
{% for star, description in stars %}
- {{star_icon.format(star)}} - {{description}}
{% endfor %}

{% endfor %}
{% endfor %}

# {{all_car_info | length}} Supported Cars

|{{Column | map(attribute='value') | join('|')}}|
|---|---|---|:---:|:---:|:---:|:---:|
{% for car_info in all_car_info %}
|{% for column in Column %}{{car_info.get_column(column, star_icon, footnote_tag)}}|{% endfor %}

{% endfor %}

<a id="footnotes"></a>
{% for footnote in footnotes %}
<sup>{{loop.index}}</sup>{{footnote}} <br />
{% endfor %}

## Community Maintained Cars
Although they're not upstream, the community has openpilot running on other makes and models. See the 'Community Supported Models' section of each make [on our wiki](https://wiki.comma.ai/).
