#include <cstdint>

#include "common.h"

namespace {

{% for address, msg_name, sigs in msgs %}
const Signal sigs_{{address}}[] = {
  {% for sig in sigs %}
    {
      {% set b1 = (sig.start_bit//8)*8  + (-sig.start_bit-1) % 8 %}
      .name = "{{sig.name}}",
      .b1 = {{b1}},
      .b2 = {{sig.size}},
      .bo = {{64 - (b1 + sig.size)}},
      .is_signed = {{"true" if sig.is_signed else "false"}},
      .factor = {{sig.factor}},
      .offset = {{sig.offset}},
      {% if checksum_type == "honda" and sig.name == "CHECKSUM" %}
      .type = SignalType::HONDA_CHECKSUM,
      {% elif checksum_type == "honda" and sig.name == "COUNTER" %}
      .type = SignalType::HONDA_COUNTER,
      {% else %}
      .type = SignalType::DEFAULT,
      {% endif %}
    },
  {% endfor %}
};
{% endfor %}

const Msg msgs[] = {
{% for address, msg_name, sigs in msgs %}
  {% set address_hex = "0x%X" % address %}
  {
    .name = "{{msg_name}}",
    .address = {{address_hex}},
    .num_sigs = ARRAYSIZE(sigs_{{address}}),
    .sigs = sigs_{{address}},
  },
{% endfor %}
};

}

const DBC {{dbc.name}} = {
  .name = "{{dbc.name}}",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
};

dbc_init({{dbc.name}})
