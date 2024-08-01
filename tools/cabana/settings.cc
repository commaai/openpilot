#include "tools/cabana/settings.h"

#include <QAbstractButton>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QPushButton>
#include <QSettings>
#include <QStandardPaths>
#include <type_traits>

#include "tools/cabana/utils/util.h"

const int MIN_CACHE_MINIUTES = 30;
const int MAX_CACHE_MINIUTES = 120;

Settings settings;

template <class SettingOperation>
void settings_op(SettingOperation op) {
  QSettings s("cabana");
  op(s, "absolute_time", settings.absolute_time);
  op(s, "fps", settings.fps);
  op(s, "max_cached_minutes", settings.max_cached_minutes);
  op(s, "chart_height", settings.chart_height);
  op(s, "chart_range", settings.chart_range);
  op(s, "chart_column_count", settings.chart_column_count);
  op(s, "last_dir", settings.last_dir);
  op(s, "last_route_dir", settings.last_route_dir);
  op(s, "window_state", settings.window_state);
  op(s, "geometry", settings.geometry);
  op(s, "video_splitter_state", settings.video_splitter_state);
  op(s, "recent_files", settings.recent_files);
  op(s, "message_header_state", settings.message_header_state);
  op(s, "chart_series_type", settings.chart_series_type);
  op(s, "theme", settings.theme);
  op(s, "sparkline_range", settings.sparkline_range);
  op(s, "multiple_lines_hex", settings.multiple_lines_hex);
  op(s, "log_livestream", settings.log_livestream);
  op(s, "log_path", settings.log_path);
  op(s, "drag_direction", (int &)settings.drag_direction);
  op(s, "suppress_defined_signals", settings.suppress_defined_signals);
}

Settings::Settings() {
  last_dir = last_route_dir = QDir::homePath();
  log_path = QStandardPaths::writableLocation(QStandardPaths::HomeLocation) + "/cabana_live_stream/";
  settings_op([](QSettings &s, const QString &key, auto &value) {
    if (auto v = s.value(key); v.canConvert<std::decay_t<decltype(value)>>())
      value = v.value<std::decay_t<decltype(value)>>();
  });
}

Settings::~Settings() {
  settings_op([](QSettings &s, const QString &key, auto &v) { s.setValue(key, v); });
}

// SettingsDlg

SettingsDlg::SettingsDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Settings"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QGroupBox *groupbox = new QGroupBox("General");
  QFormLayout *form_layout = new QFormLayout(groupbox);

  form_layout->addRow(tr("Color Theme"), theme = new QComboBox(this));
  theme->setToolTip(tr("You may need to restart cabana after changes theme"));
  theme->addItems({tr("Automatic"), tr("Light"), tr("Dark")});
  theme->setCurrentIndex(settings.theme);

  form_layout->addRow("FPS", fps = new QSpinBox(this));
  fps->setRange(10, 100);
  fps->setSingleStep(10);
  fps->setValue(settings.fps);

  form_layout->addRow(tr("Max Cached Minutes"), cached_minutes = new QSpinBox(this));
  cached_minutes->setRange(MIN_CACHE_MINIUTES, MAX_CACHE_MINIUTES);
  cached_minutes->setSingleStep(1);
  cached_minutes->setValue(settings.max_cached_minutes);
  main_layout->addWidget(groupbox);

  groupbox = new QGroupBox("New Signal Settings");
  form_layout = new QFormLayout(groupbox);
  form_layout->addRow(tr("Drag Direction"), drag_direction = new QComboBox(this));
  drag_direction->addItems({tr("MSB First"), tr("LSB First"), tr("Always Little Endian"), tr("Always Big Endian")});
  drag_direction->setCurrentIndex(settings.drag_direction);
  main_layout->addWidget(groupbox);

  groupbox = new QGroupBox("Chart");
  form_layout = new QFormLayout(groupbox);
  form_layout->addRow(tr("Default Series Type"), chart_series_type = new QComboBox(this));
  chart_series_type->addItems({tr("Line"), tr("Step Line"), tr("Scatter")});
  chart_series_type->setCurrentIndex(settings.chart_series_type);

  form_layout->addRow(tr("Chart Height"), chart_height = new QSpinBox(this));
  chart_height->setRange(100, 500);
  chart_height->setSingleStep(10);
  chart_height->setValue(settings.chart_height);
  main_layout->addWidget(groupbox);

  log_livestream = new QGroupBox(tr("Enable live stream logging"), this);
  log_livestream->setCheckable(true);
  QHBoxLayout *path_layout = new QHBoxLayout(log_livestream);
  path_layout->addWidget(log_path = new QLineEdit(settings.log_path, this));
  log_path->setReadOnly(true);
  auto browse_btn = new QPushButton(tr("B&rowse..."));
  path_layout->addWidget(browse_btn);
  main_layout->addWidget(log_livestream);

  auto buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  main_layout->addWidget(buttonBox);
  setFixedSize(400, sizeHint().height());

  QObject::connect(browse_btn, &QPushButton::clicked, [this]() {
    QString fn = QFileDialog::getExistingDirectory(
        this, tr("Log File Location"),
        QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!fn.isEmpty()) {
      log_path->setText(fn);
    }
  });
  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &SettingsDlg::save);
}

void SettingsDlg::save() {
  if (std::exchange(settings.theme, theme->currentIndex()) != settings.theme) {
    // set theme before emit changed
    utils::setTheme(settings.theme);
  }
  settings.fps = fps->value();
  settings.max_cached_minutes = cached_minutes->value();
  settings.chart_series_type = chart_series_type->currentIndex();
  settings.chart_height = chart_height->value();
  settings.log_livestream = log_livestream->isChecked();
  settings.log_path = log_path->text();
  settings.drag_direction = (Settings::DragDirection)drag_direction->currentIndex();
  emit settings.changed();
  QDialog::accept();
}
