#include "tools/cabana/route.h"

#include <QFileDialog>
#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRadioButton>

#include "tools/cabana/streams/replaystream.h"

OpenRouteDialog::OpenRouteDialog(QWidget *parent) : QDialog(parent) {
  // TODO: get route list from api.comma.ai
  QGridLayout *grid_layout = new QGridLayout();
  grid_layout->addWidget(new QLabel(tr("Route")), 0, 0);
  grid_layout->addWidget(route_edit = new QLineEdit(this), 0, 1);
  route_edit->setPlaceholderText(tr("Enter remote route name or click browse to select a local route"));
  auto file_btn = new QPushButton(tr("Browse..."), this);
  grid_layout->addWidget(file_btn, 0, 2);

  QHBoxLayout *video_btn_layout = new QHBoxLayout();
  video_btn_group = new QButtonGroup(this);
  QString buttons[] = {tr("No Video"), tr("Road"), tr("Wide Road"), tr("Driver"), tr("QCamera")};
  for (int i = 0; i < std::size(buttons); ++i) {
    auto btn = new QRadioButton(buttons[i], this);
    btn->setChecked(i == 1);  // default is road camera
    video_btn_group->addButton(btn, i);
    video_btn_layout->addWidget(btn);
  }
  video_btn_layout->addStretch(1);
  grid_layout->addLayout(video_btn_layout, 1, 1, 1, 2);

  btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
  btn_box->button(QDialogButtonBox::Open)->setEnabled(false);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(grid_layout);
  main_layout->addWidget(btn_box);
  main_layout->addStretch(1);

  QObject::connect(btn_box, &QDialogButtonBox::accepted, this, &OpenRouteDialog::loadRoute);
  QObject::connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(route_edit, &QLineEdit::textChanged, [this]() {
    btn_box->button(QDialogButtonBox::Open)->setEnabled(!route_edit->text().isEmpty());
  });
  QObject::connect(file_btn, &QPushButton::clicked, [=]() {
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Local Route"), settings.last_route_dir);
    if (!dir.isEmpty()) {
      route_edit->setText(dir);
      settings.last_route_dir = QFileInfo(dir).absolutePath();
    }
  });
}

void OpenRouteDialog::loadRoute() {
  btn_box->setEnabled(false);

  QString route = route_edit->text();
  QString data_dir;
  if (int idx = route.lastIndexOf('/'); idx != -1) {
    data_dir = route.mid(0, idx + 1);
    route = route.mid(idx + 1);
  }

  bool is_valid_format = Route::parseRoute(route).str.size() > 0;
  if (!is_valid_format) {
    QMessageBox::warning(nullptr, tr("Warning"), tr("Invalid route format: '%1'").arg(route));
  } else {
    uint32_t flags[] = {REPLAY_FLAG_NO_VIPC, REPLAY_FLAG_NONE, REPLAY_FLAG_ECAM, REPLAY_FLAG_DCAM, REPLAY_FLAG_QCAMERA};
    failed_to_load = !dynamic_cast<ReplayStream *>(can)->loadRoute(route, data_dir, flags[video_btn_group->checkedId()]);
    if (failed_to_load) {
      QMessageBox::warning(nullptr, tr("Warning"), tr("Failed to load route: '%1'").arg(route));
    } else {
      accept();
    }
  }

  btn_box->setEnabled(true);
}
