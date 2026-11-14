#pragma once

#include <algorithm>
#include <optional>
#include <vector>

#include "tools/cabana/dbc/dbc.h"

// Only the current multiplexed branch owns the payload bits. Use this same
// signal set for painting, endpoint markers, and mouse interaction.
inline std::vector<const cabana::Signal *> binaryViewSignals(const cabana::Msg *msg, const uint8_t *data, size_t size) {
  if (!msg) return {};
  std::optional<double> multiplex_value;
  if (msg->multiplexor) {
    auto selector = *msg->multiplexor;
    if (std::max(selector.msb, selector.lsb) / 8 < size) {
      // Multiplex values are raw identifiers, independent of display scaling.
      selector.factor = 1;
      selector.offset = 0;
      selector.is_signed = false;
      multiplex_value = get_raw_value(data, size, selector);
    }
  }
  std::vector<const cabana::Signal *> signals;
  for (auto sig : msg->getSignals()) {
    if (sig->type != cabana::Signal::Type::Multiplexed ||
        (multiplex_value && *multiplex_value == sig->multiplex_value)) {
      signals.push_back(sig);
    }
  }
  return signals;
}
