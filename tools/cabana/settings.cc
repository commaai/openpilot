#include "tools/cabana/settings.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QSettings>

// Settings
Settings settings;

Settings::Settings() {
  load();
}

void Settings::save() {
  QSettings s("settings", QSettings::IniFormat);
  s.setValue("fps", fps);
  s.setValue("log_size", can_msg_log_size);
  s.setValue("cached_segment", cached_segment_limit);
  s.setValue("chart_height", chart_height);
  emit changed();
}

void Settings::load() {
  QSettings s("settings", QSettings::IniFormat);
  fps = s.value("fps", 10).toInt();
  can_msg_log_size = s.value("log_size", 100).toInt();
  cached_segment_limit = s.value("cached_segment", 3).toInt();
  chart_height = s.value("chart_height", 200).toInt();
}

// SettingsDlg

SettingsDlg::SettingsDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Settings"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QFormLayout *form_layout = new QFormLayout();

  fps = new QSpinBox(this);
  fps->setRange(10, 100);
  fps->setSingleStep(10);
  fps->setValue(settings.fps);
  form_layout->addRow("FPS", fps);

  log_size = new QSpinBox(this);
  log_size->setRange(50, 500);
  log_size->setSingleStep(10);
  log_size->setValue(settings.can_msg_log_size);
  form_layout->addRow(tr("Log size"), log_size);

  cached_segment = new QSpinBox(this);
  cached_segment->setRange(3, 60);
  cached_segment->setSingleStep(1);
  cached_segment->setValue(settings.cached_segment_limit);
  form_layout->addRow(tr("Cached segments limit"), cached_segment);

  chart_height = new QSpinBox(this);
  chart_height->setRange(100, 500);
  chart_height->setSingleStep(10);
  chart_height->setValue(settings.chart_height);
  form_layout->addRow(tr("Chart height"), chart_height);

  main_layout->addLayout(form_layout);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);

  setFixedWidth(360);
  connect(buttonBox, &QDialogButtonBox::accepted, this, &SettingsDlg::save);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void SettingsDlg::save() {
  settings.fps = fps->value();
  settings.can_msg_log_size = log_size->value();
  settings.cached_segment_limit = cached_segment->value();
  settings.chart_height = chart_height->value();
  settings.save();
  accept();
}
