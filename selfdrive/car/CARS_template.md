{% set footnote_tag = '[<sup>{}</sup>](#footnotes)' -%}
{% set star_icon = '<a href="##"><img valign="top" src="assets/icon-star-{}.svg" width="22" /></a>' -%}

# Supported Cars

A supported vehicle is one that just works when you install a comma device. Every car performs differently with openpilot, but all supported cars should provide a better experience than any stock system.

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
