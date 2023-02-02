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
  int max_cached_minutes = 5;
  int chart_height = 200;
  int chart_column_count = 1;
  int chart_range = 3 * 60; // e minutes
  int chart_series_type = 0;
  QString last_dir;
  QByteArray geometry;
  QByteArray video_splitter_state;
  QByteArray window_state;
  QStringList recent_files;
  QByteArray message_header_state;

signals:
  void changed();
};

class SettingsDlg : public QDialog {
  Q_OBJECT

public:
  SettingsDlg(QWidget *parent);
  void save();
  QSpinBox *fps;
  QSpinBox *cached_minutes;
  QSpinBox *chart_height;
  QComboBox *chart_series_type;
};

extern Settings settings;
