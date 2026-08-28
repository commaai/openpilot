#pragma once

#include <cstdint>
#include <vector>

#include "tools/cabana/core/observable.h"
#include "tools/cabana/core/settings.h"

class Settings : public CabanaSettingsState {
public:
  Settings();
  void save();

  // Qt frontend layout state. This intentionally stays outside CabanaSettingsState.
  std::vector<uint8_t> geometry;
  std::vector<uint8_t> video_splitter_state;
  std::vector<uint8_t> window_state;
  std::vector<uint8_t> message_header_state;

  Observable<> changed;
};

extern Settings settings;
