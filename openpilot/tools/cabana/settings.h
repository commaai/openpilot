#pragma once

#include <cstdint>
#include <vector>

#include <QComboBox>
#include <QDialog>
#include <QGroupBox>
#include <QLineEdit>
#include <QSpinBox>

#include "tools/cabana/core/settings.h"

class Settings : public QObject, public CabanaSettingsState {
  Q_OBJECT

public:
  Settings();
  void save();

  // Qt frontend layout state. This intentionally stays outside CabanaSettingsState.
  std::vector<uint8_t> geometry;
  std::vector<uint8_t> video_splitter_state;
  std::vector<uint8_t> window_state;
  std::vector<uint8_t> message_header_state;

signals:
  void changed();
};

class SettingsDlg : public QDialog {
public:
  SettingsDlg(QWidget *parent);
  void save();
  QSpinBox *fps;
  QSpinBox *cached_minutes;
  QSpinBox *chart_height;
  QComboBox *chart_series_type;
  QComboBox *theme;
  QGroupBox *log_livestream;
  QLineEdit *log_path;
  QComboBox *drag_direction;
};

extern Settings settings;
