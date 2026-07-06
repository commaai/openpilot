#pragma once

#include "tools/loggy/backend/live.h"
#include "tools/loggy/backend/session.h"

#include <array>
#include <string>

namespace loggy {

struct LiveSourceUiState {
  bool open_popup = false;
  std::array<char, 128> address_buffer{};
  int source_kind_index = 0;
  std::array<int, kPandaBusCount> panda_can_speed_index{};
  std::array<int, kPandaBusCount> panda_data_speed_index{};
  std::array<bool, kPandaBusCount> panda_can_fd{};
  double buffer_seconds = 30.0;
  std::string status;
};

int live_source_kind_to_index(LiveSourceKind kind);

void sync_live_source_fields(const Session &session, LiveSourceUiState &state);
void request_live_source_popup(const Session &session, LiveSourceUiState &state);
void draw_live_source_popup(Session &session, LiveSourceUiState &state);

}  // namespace loggy
