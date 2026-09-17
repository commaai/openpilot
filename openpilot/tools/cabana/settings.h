#pragma once

#include <string>

#include "tools/cabana/core/observable.h"
#include "tools/cabana/core/settings.h"

class Settings : public CabanaSettingsState {
public:
  Settings();
  void save();

  // UI layout state (dock layout, window geometry, table state), owned by the imgui frontend
  std::string ui_state;

  Observable<> changed;
};

extern Settings settings;
