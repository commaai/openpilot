#include "common_dbc.h"

namespace {

{% for address, msg_name, msg_size, sigs in msgs %}
const Signal sigs_{{address}}[] = {
  {% for sig in sigs %}
    {
      {% if sig.is_little_endian %}
        {% set b1 = sig.start_bit %}
      {% else %}
        {% set b1 = (sig.start_bit//8)*8  + (-sig.start_bit-1) % 8 %}
      {% endif %}
      .name = "{{sig.name}}",
      .b1 = {{b1}},
      .b2 = {{sig.size}},
      .bo = {{64 - (b1 + sig.size)}},
      .is_signed = {{"true" if sig.is_signed else "false"}},
      .factor = {{sig.factor}},
      .offset = {{sig.offset}},
      .is_little_endian = {{"true" if sig.is_little_endian else "false"}},
      {% if checksum_type == "honda" and sig.name == "CHECKSUM" %}
      .type = SignalType::HONDA_CHECKSUM,
      {% elif checksum_type == "honda" and sig.name == "COUNTER" %}
      .type = SignalType::HONDA_COUNTER,
      {% elif checksum_type == "toyota" and sig.name == "CHECKSUM" %}
      .type = SignalType::TOYOTA_CHECKSUM,
      {% elif checksum_type == "volkswagen" and sig.name == "CHECKSUM" %}
      .type = SignalType::VOLKSWAGEN_CHECKSUM,
      {% elif checksum_type == "volkswagen" and sig.name == "COUNTER" %}
      .type = SignalType::VOLKSWAGEN_COUNTER,
      {% elif address in [512, 513] and sig.name == "CHECKSUM_PEDAL" %}
      .type = SignalType::PEDAL_CHECKSUM,
      {% elif address in [512, 513] and sig.name == "COUNTER_PEDAL" %}
      .type = SignalType::PEDAL_COUNTER,
      {% else %}
      .type = SignalType::DEFAULT,
      {% endif %}
    },
  {% endfor %}
};
{% endfor %}

const Msg msgs[] = {
{% for address, msg_name, msg_size, sigs in msgs %}
  {% set address_hex = "0x%X" % address %}
  {
    .name = "{{msg_name}}",
    .address = {{address_hex}},
    .size = {{msg_size}},
    .num_sigs = ARRAYSIZE(sigs_{{address}}),
    .sigs = sigs_{{address}},
  },
{% endfor %}
};

const Val vals[] = {
{% for address, sig in def_vals %}
  {% for sg_name, def_val in sig %}
    {% set address_hex = "0x%X" % address %}
    {
      .name = "{{sg_name}}",
      .address = {{address_hex}},
      .def_val = {{def_val}},
      .sigs = sigs_{{address}},
    },
  {% endfor %}
{% endfor %}
};

}

const DBC {{dbc.name}} = {
  .name = "{{dbc.name}}",
  .num_msgs = ARRAYSIZE(msgs),
  .msgs = msgs,
  .vals = vals,
  .num_vals = ARRAYSIZE(vals),
};

dbc_init({{dbc.name}})
