// Supported Vehicles Vuex Store
// ~~~~~~~~~~~~~~~
{% set footnote_tag = '<a style="position: absolute;" href="/vehicles/#footnote"><sup>{}</sup></a>' -%}
{% set star_icon = '<img src="/supported-cars/icon-star-{}.svg" alt="">' -%}

import axios from 'axios';

export const state = () => ({
  leverJobs: [],
  columns: [
    {% for column in Column %}
    '{{column.value}}',
    {% endfor %}
  ],
  footnotes: [
    {% for footnote in footnotes %}
    '{{footnote | replace("'", "\\'")}}',
    {% endfor %}
  ],
  supportedVehicles: {
    {% for tier, cars in tiers %}
    '{{tier.name.title()}}': {
      description: '{{tier.value | replace("'", "\\'")}}',
      cars: [
        {% for car_info in cars %}
        {
          video_link: '{{car_info.video_link}}',
          row: {
            {% for column in Column %}
            {{column.name.lower()}}: '{{car_info.get_column(column, star_icon, footnote_tag)}}',
            {% endfor %}
          }
        },
        {% endfor %}
      ],
    },
    {% endfor %}
  },
})

export const mutations = {}

export const actions = {}
