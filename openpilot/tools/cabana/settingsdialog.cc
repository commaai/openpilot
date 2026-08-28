#include "tools/cabana/settingsdialog.h"

#include <utility>

#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QPushButton>
#include <QVBoxLayout>

#include "tools/cabana/settings.h"
#include "tools/cabana/utils/qtutil.h"

const int MIN_CACHE_MINIUTES = 30;
const int MAX_CACHE_MINIUTES = 120;

SettingsDialog::SettingsDialog(QWidget *parent) : QDialog(parent) {
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
  form_layout->addRow(tr("Chart Height"), chart_height = new QSpinBox(this));
  chart_height->setRange(100, 500);
  chart_height->setSingleStep(10);
  chart_height->setValue(settings.chart_height);
  main_layout->addWidget(groupbox);

  log_livestream = new QGroupBox(tr("Enable live stream logging"), this);
  log_livestream->setCheckable(true);
  log_livestream->setChecked(settings.log_livestream);
  QHBoxLayout *path_layout = new QHBoxLayout(log_livestream);
  path_layout->addWidget(log_path = new QLineEdit(QString::fromStdString(settings.log_path), this));
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
        QString::fromStdString(utils::homePath()),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (!fn.isEmpty()) {
      log_path->setText(fn);
    }
  });
  QObject::connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(buttonBox, &QDialogButtonBox::accepted, this, &SettingsDialog::save);
}

void SettingsDialog::save() {
  if (std::exchange(settings.theme, theme->currentIndex()) != settings.theme) {
    // set theme before emit changed
    utils::setTheme(settings.theme);
  }
  settings.fps = fps->value();
  settings.max_cached_minutes = cached_minutes->value();
  settings.chart_height = chart_height->value();
  settings.log_livestream = log_livestream->isChecked();
  settings.log_path = log_path->text().toStdString();
  settings.drag_direction = (Settings::DragDirection)drag_direction->currentIndex();
  settings.changed();
  QDialog::accept();
}
