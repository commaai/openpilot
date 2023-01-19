#pragma once

#include <QByteArray>
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
  int cached_segment_limit = 3;
  int chart_height = 200;
  int max_chart_x_range = 3 * 60; // 3 minutes
  QString last_dir;
  QByteArray geometry;
  QByteArray video_splitter_state;
  QByteArray window_state;

signals:
  void changed();
};

class SettingsDlg : public QDialog {
  Q_OBJECT

public:
  SettingsDlg(QWidget *parent);
  void save();
  QSpinBox *fps;
  QSpinBox *cached_segment;
  QSpinBox *chart_height;
  QSpinBox *max_chart_x_range;
};

extern Settings settings;
