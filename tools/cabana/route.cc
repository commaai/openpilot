#include "tools/cabana/route.h"

#include <QButtonGroup>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>

#include "tools/cabana/streams/replaystream.h"

// OpenRouteDialog

OpenRouteDialog::OpenRouteDialog(QWidget *parent) : QDialog(parent) {
  QHBoxLayout *edit_layout = new QHBoxLayout;
  edit_layout->addWidget(new QLabel(tr("Route:")));
  route_edit = new QLineEdit(this);
  route_edit->setPlaceholderText(tr("Enter remote route name or click browse to select a local route"));
  edit_layout->addWidget(route_edit);
  auto file_btn = new QPushButton(tr("Browse..."), this);
  edit_layout->addWidget(file_btn);

  btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->addStretch(0);
  main_layout->addLayout(edit_layout);
  main_layout->addStretch(0);
  main_layout->addWidget(btn_box);
  setMinimumSize({550, 120});

  QObject::connect(btn_box, &QDialogButtonBox::rejected, [this]() { success ? accept() : reject(); });
  QObject::connect(btn_box, &QDialogButtonBox::accepted, this, &OpenRouteDialog::loadRoute);
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
    QString basename = route.mid(idx + 1);
    if (int pos = basename.lastIndexOf("--"); pos != -1) {
      data_dir = route.mid(0, idx);
      route = basename.mid(0, pos);
    }
  }

  ReplayStream *replay_stream = dynamic_cast<ReplayStream *>(can);
  success = replay_stream->loadRoute(route, data_dir);
  if (success) {
    accept();
  } else {
    QString text = tr("Failed to load route: '%1'").arg(route);
    QMessageBox::warning(nullptr, tr("Warning"), text);
  }
  btn_box->setEnabled(true);
}
