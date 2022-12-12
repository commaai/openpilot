#include "tools/cabana/settings.h"

#include <QDialogButtonBox>
#include <QDir>
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
  s.setValue("cached_segment", cached_segment_limit);
  s.setValue("chart_height", chart_height);
  s.setValue("max_chart_x_range", max_chart_x_range);
  s.setValue("last_dir", last_dir);
  s.setValue("splitter_state", splitter_state);
}

void Settings::load() {
  QSettings s("settings", QSettings::IniFormat);
  fps = s.value("fps", 10).toInt();
  cached_segment_limit = s.value("cached_segment", 3).toInt();
  chart_height = s.value("chart_height", 200).toInt();
  max_chart_x_range = s.value("max_chart_x_range", 3 * 60).toInt();
  last_dir = s.value("last_dir", QDir::homePath()).toString();
  splitter_state = s.value("splitter_state").toByteArray();
}

// SettingsDlg

SettingsDlg::SettingsDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Settings"));
  QFormLayout *form_layout = new QFormLayout(this);

  fps = new QSpinBox(this);
  fps->setRange(10, 100);
  fps->setSingleStep(10);
  fps->setValue(settings.fps);
  form_layout->addRow("FPS", fps);

  cached_segment = new QSpinBox(this);
  cached_segment->setRange(3, 60);
  cached_segment->setSingleStep(1);
  cached_segment->setValue(settings.cached_segment_limit);
  form_layout->addRow(tr("Cached segments limit"), cached_segment);

  max_chart_x_range = new QSpinBox(this);
  max_chart_x_range->setRange(1, 60);
  max_chart_x_range->setSingleStep(1);
  max_chart_x_range->setValue(settings.max_chart_x_range / 60);
  form_layout->addRow(tr("Chart range (minutes)"), max_chart_x_range);

  chart_height = new QSpinBox(this);
  chart_height->setRange(100, 500);
  chart_height->setSingleStep(10);
  chart_height->setValue(settings.chart_height);
  form_layout->addRow(tr("Chart height"), chart_height);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  form_layout->addRow(buttonBox);

  setFixedWidth(360);
  connect(buttonBox, &QDialogButtonBox::accepted, this, &SettingsDlg::save);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void SettingsDlg::save() {
  settings.fps = fps->value();
  settings.cached_segment_limit = cached_segment->value();
  settings.chart_height = chart_height->value();
  settings.max_chart_x_range = max_chart_x_range->value() * 60;
  settings.save();
  accept();
  emit settings.changed();
}
