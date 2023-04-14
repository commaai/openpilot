#include "tools/cabana/route.h"

#include <QFileDialog>
#include <QGridLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>

#include "tools/cabana/streams/replaystream.h"

OpenRouteDialog::OpenRouteDialog(QWidget *parent) : QDialog(parent) {
  // TODO: get route list from api.comma.ai
  QGridLayout *grid_layout = new QGridLayout();
  grid_layout->addWidget(new QLabel(tr("Route")), 0, 0);
  grid_layout->addWidget(route_edit = new QLineEdit(this), 0, 1);
  route_edit->setPlaceholderText(tr("Enter remote route name or click browse to select a local route"));
  auto file_btn = new QPushButton(tr("Browse..."), this);
  grid_layout->addWidget(file_btn, 0, 2);

  grid_layout->addWidget(new QLabel(tr("Video")), 1, 0);
  grid_layout->addWidget(choose_video_cb = new QComboBox(this), 1, 1);
  QString items[] = {tr("No Video"), tr("Road Camera"), tr("Wide Road Camera"), tr("Driver Camera"), tr("QCamera")};
  for (int i = 0; i < std::size(items); ++i) {
    choose_video_cb->addItem(items[i]);
  }
  choose_video_cb->setCurrentIndex(1);  // default is road camera;

  btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
  btn_box->button(QDialogButtonBox::Open)->setEnabled(false);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addLayout(grid_layout);
  main_layout->addWidget(btn_box);
  main_layout->addStretch(0);
  setMinimumSize({550, 120});

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
    failed_to_load = !dynamic_cast<ReplayStream *>(can)->loadRoute(route, data_dir, flags[choose_video_cb->currentIndex()]);
    if (failed_to_load) {
      QMessageBox::warning(nullptr, tr("Warning"), tr("Failed to load route: '%1'").arg(route));
    } else {
      accept();
    }
  }

  btn_box->setEnabled(true);
}
