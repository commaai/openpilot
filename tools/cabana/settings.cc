#include "tools/cabana/settings.h"

#include <QAbstractButton>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QPushButton>
#include <QStandardPaths>

#include "tools/cabana/util.h"

Settings settings;

QSettings::Status Settings::save() {
  QSettings s(filePath(), QSettings::IniFormat);
  s.setValue("absolute_time", absolute_time);
  s.setValue("fps", fps);
  s.setValue("max_cached_minutes", max_cached_minutes);
  s.setValue("chart_height", chart_height);
  s.setValue("chart_range", chart_range);
  s.setValue("chart_column_count", chart_column_count);
  s.setValue("last_dir", last_dir);
  s.setValue("last_route_dir", last_route_dir);
  s.setValue("window_state", window_state);
  s.setValue("geometry", geometry);
  s.setValue("video_splitter_state", video_splitter_state);
  s.setValue("recent_files", recent_files);
  s.setValue("message_header_state_v3", message_header_state);
  s.setValue("chart_series_type", chart_series_type);
  s.setValue("theme", theme);
  s.setValue("sparkline_range", sparkline_range);
  s.setValue("multiple_lines_bytes", multiple_lines_bytes);
  s.setValue("log_livestream", log_livestream);
  s.setValue("log_path", log_path);
  s.setValue("drag_direction", drag_direction);
  s.setValue("suppress_defined_signals", suppress_defined_signals);
  s.sync();
  return s.status();
}

void Settings::load() {
  QSettings s(filePath(), QSettings::IniFormat);
  absolute_time = s.value("absolute_time", false).toBool();
  fps = s.value("fps", 10).toInt();
  max_cached_minutes = s.value("max_cached_minutes", 30).toInt();
  chart_height = s.value("chart_height", 200).toInt();
  chart_range = s.value("chart_range", 3 * 60).toInt();
  chart_column_count = s.value("chart_column_count", 1).toInt();
  last_dir = s.value("last_dir", QDir::homePath()).toString();
  last_route_dir = s.value("last_route_dir", QDir::homePath()).toString();
  window_state = s.value("window_state").toByteArray();
  geometry = s.value("geometry").toByteArray();
  video_splitter_state = s.value("video_splitter_state").toByteArray();
  recent_files = s.value("recent_files").toStringList();
  message_header_state = s.value("message_header_state_v3").toByteArray();
  chart_series_type = s.value("chart_series_type", 0).toInt();
  theme = s.value("theme", 0).toInt();
  sparkline_range = s.value("sparkline_range", 15).toInt();
  multiple_lines_bytes = s.value("multiple_lines_bytes", true).toBool();
  log_livestream = s.value("log_livestream", true).toBool();
  log_path = s.value("log_path").toString();
  drag_direction = (Settings::DragDirection)s.value("drag_direction", 0).toInt();
  suppress_defined_signals = s.value("suppress_defined_signals", false).toBool();
  if (log_path.isEmpty()) {
    log_path = QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/cabana_live_stream/";
  }
}

// SettingsDlg

SettingsDlg::SettingsDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Settings"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QGroupBox *groupbox = new QGroupBox("General");
  QFormLayout *form_layout = new QFormLayout(groupbox);

  theme = new QComboBox(this);
  theme->setToolTip(tr("You may need to restart cabana after changes theme"));
  theme->addItems({tr("Automatic"), tr("Light"), tr("Dark")});
  theme->setCurrentIndex(settings.theme);
  form_layout->addRow(tr("Color Theme"), theme);

  fps = new QSpinBox(this);
  fps->setRange(10, 100);
  fps->setSingleStep(10);
  fps->setValue(settings.fps);
  form_layout->addRow("FPS", fps);

  cached_minutes = new QSpinBox(this);
  cached_minutes->setRange(5, 60);
  cached_minutes->setSingleStep(1);
  cached_minutes->setValue(settings.max_cached_minutes);
  form_layout->addRow(tr("Max Cached Minutes"), cached_minutes);
  main_layout->addWidget(groupbox);

  groupbox = new QGroupBox("New Signal Settings");
  form_layout = new QFormLayout(groupbox);
  drag_direction = new QComboBox(this);
  drag_direction->addItems({tr("MSB First"), tr("LSB First"), tr("Always Little Endian"), tr("Always Big Endian")});
  drag_direction->setCurrentIndex(settings.drag_direction);
  form_layout->addRow(tr("Drag Direction"), drag_direction);
  main_layout->addWidget(groupbox);

  groupbox = new QGroupBox("Chart");
  form_layout = new QFormLayout(groupbox);
  chart_series_type = new QComboBox(this);
  chart_series_type->addItems({tr("Line"), tr("Step Line"), tr("Scatter")});
  chart_series_type->setCurrentIndex(settings.chart_series_type);
  form_layout->addRow(tr("Chart Default Series Type"), chart_series_type);

  chart_height = new QSpinBox(this);
  chart_height->setRange(100, 500);
  chart_height->setSingleStep(10);
  chart_height->setValue(settings.chart_height);
  form_layout->addRow(tr("Chart Height"), chart_height);
  main_layout->addWidget(groupbox);

  log_livestream = new QGroupBox(tr("Enable live stream logging"), this);
  log_livestream->setCheckable(true);
  QHBoxLayout *path_layout = new QHBoxLayout(log_livestream);
  path_layout->addWidget(log_path = new QLineEdit(settings.log_path, this));
  log_path->setReadOnly(true);
  auto browse_btn = new QPushButton(tr("B&rowse..."));
  path_layout->addWidget(browse_btn);
  main_layout->addWidget(log_livestream);


  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Apply);
  main_layout->addWidget(buttonBox);
  main_layout->addStretch(1);

  QObject::connect(browse_btn, &QPushButton::clicked, [this]() {
    QString fn = QFileDialog::getExistingDirectory(
        this, tr("Log File Location"),
        QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!fn.isEmpty()) {
      log_path->setText(fn);
    }
  });
  QObject::connect(buttonBox, &QDialogButtonBox::clicked, [=](QAbstractButton *button) {
    auto role = buttonBox->buttonRole(button);
    if (role == QDialogButtonBox::AcceptRole) {
      save();
      accept();
    } else if (role == QDialogButtonBox::ApplyRole) {
      save();
    } else if (role == QDialogButtonBox::RejectRole) {
      reject();
    }
  });
}

void SettingsDlg::save() {
  settings.fps = fps->value();
  if (std::exchange(settings.theme, theme->currentIndex()) != settings.theme) {
    // set theme before emit changed
    utils::setTheme(settings.theme);
  }
  settings.max_cached_minutes = cached_minutes->value();
  settings.chart_series_type = chart_series_type->currentIndex();
  settings.chart_height = chart_height->value();
  settings.log_livestream = log_livestream->isChecked();
  settings.log_path = log_path->text();
  settings.drag_direction = (Settings::DragDirection)drag_direction->currentIndex();
  settings.save();
  emit settings.changed();
}
