#pragma once

#include <QComboBox>
#include <QDialog>
#include <QSpinBox>

class Settings : public QObject {
  Q_OBJECT

public:
  Settings();
  void save();
  void load();

  int fps = 10;
  int can_msg_log_size = 100;
  int cached_segment_limit = 3;
  int chart_height = 200;
  int chart_theme = 0;

signals:
  void changed();
};

class SettingsDlg : public QDialog {
  Q_OBJECT

public:
  SettingsDlg(QWidget *parent);
  void save();
  QSpinBox *fps;
  QSpinBox *log_size ;
  QSpinBox *cached_segment;
  QSpinBox *chart_height;
  QComboBox *chart_theme;
};

extern Settings settings;
