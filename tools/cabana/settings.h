#pragma once

#include <QApplication>
#include <QByteArray>
#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QGroupBox>
#include <QLineEdit>
#include <QSettings>
#include <QSpinBox>

#define LIGHT_THEME 1
#define DARK_THEME 2

class Settings : public QObject {
  Q_OBJECT

public:
  enum DragDirection {
    MsbFirst,
    LsbFirst,
    AlwaysLE,
    AlwaysBE,
  };

  Settings() {}
  QSettings::Status save();
  void load();
  inline static QString filePath() { return QApplication::applicationDirPath() + "/settings"; }

  bool absolute_time = false;
  int fps = 10;
  int max_cached_minutes = 30;
  int chart_height = 200;
  int chart_column_count = 1;
  int chart_range = 3 * 60; // 3 minutes
  int chart_series_type = 0;
  int theme = 0;
  int sparkline_range = 15; // 15 seconds
  bool multiple_lines_bytes = true;
  bool log_livestream = true;
  bool suppress_defined_signals = false;
  QString log_path;
  QString last_dir;
  QString last_route_dir;
  QByteArray geometry;
  QByteArray video_splitter_state;
  QByteArray window_state;
  QStringList recent_files;
  QByteArray message_header_state;
  DragDirection drag_direction;

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
  QComboBox *theme;
  QGroupBox *log_livestream;
  QLineEdit *log_path;
  QComboBox *drag_direction;
};

extern Settings settings;
