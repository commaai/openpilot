#pragma once

#include <QByteArray>
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
  QByteArray geometry;
  QByteArray video_splitter_state;
  QByteArray window_state;
  QByteArray message_header_state;

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
