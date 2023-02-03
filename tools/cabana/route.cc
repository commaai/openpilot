#include "tools/cabana/route.h"

#include <QButtonGroup>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>

#include "tools/cabana/streams/replaystream.h"

OpenRouteDialog::OpenRouteDialog(QWidget *parent) : QDialog(parent) {
  // TODO: get route list from api.comma.ai
  QHBoxLayout *edit_layout = new QHBoxLayout;
  edit_layout->addWidget(new QLabel(tr("Route:")));
  edit_layout->addWidget(route_edit = new QLineEdit(this));
  route_edit->setPlaceholderText(tr("Enter remote route name or click browse to select a local route"));
  auto file_btn = new QPushButton(tr("Browse..."), this);
  edit_layout->addWidget(file_btn);

  btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
  btn_box->button(QDialogButtonBox::Open)->setEnabled(false);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addStretch(0);
  main_layout->addLayout(edit_layout);
  main_layout->addStretch(0);
  main_layout->addWidget(btn_box);
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
    failed_to_load = !dynamic_cast<ReplayStream *>(can)->loadRoute(route, data_dir);
    if (failed_to_load) {
      QMessageBox::warning(nullptr, tr("Warning"), tr("Failed to load route: '%1'").arg(route));
    } else {
      accept();
    }
  }

  btn_box->setEnabled(true);
}
