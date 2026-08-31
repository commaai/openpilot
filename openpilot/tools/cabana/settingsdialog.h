#pragma once

#include <QComboBox>
#include <QDialog>
#include <QGroupBox>
#include <QLineEdit>
#include <QSpinBox>

class SettingsDialog : public QDialog {
public:
  SettingsDialog(QWidget *parent);
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
