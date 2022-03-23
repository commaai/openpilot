// Supported Vehicles Vuex Store
// ~~~~~~~~~~~~~~~
{% set footnote_tag = '<a style="position: absolute;" href="/vehicles/#footnote"><sup>{}</sup></a>' %}

import axios from 'axios';

export const state = () => ({
  leverJobs: [],
  columns: [
    {% for column in columns %}
    '{{column}}',
    {% endfor %}
  ],
  footnotes: [
    {% for footnote in footnotes %}
    '{{ footnote | replace("'", "\\'") }}',
    {% endfor %}
  ],
  supportedVehicles: {
    {% for tier, car_rows in tiers %}
    '{{tier.name.title()}}': {
      description: '{{ tier.value | replace("'", "\\'") }}',
      rows: [
        {% for row in car_rows %}
        [
          '{{row[0].text}}',
          '{{row[1].text}}',
          '{{row[2].text}}',
          {% for star_col in row if star_col.star is not none %}
          '{{star_col.star.html_icon}}{{footnote_tag.format(star_col.footnote) if star_col.footnote else ''}}',
          {% endfor %}
        ],
        {% endfor %}
      ],
    },
    {% endfor %}
  },
})

export const mutations = {}

export const actions = {}
