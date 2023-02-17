#include "tools/cabana/route.h"

#include <QFileDialog>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QMessageBox>

#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/util.h"
#include "tools/cabana/streams/replaystream.h"

OpenRouteDialog::OpenRouteDialog(QWidget *parent) : QDialog(parent) {
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
  main_layout->addWidget(btn_box);
  QFrame *hline = new QFrame();
  hline->setFrameStyle(QFrame::HLine | QFrame::Sunken);
  main_layout->addWidget(hline);
  main_layout->addWidget(route_list = new RemoteRouteList(this));

  setMinimumSize({550, 120});

  QObject::connect(btn_box, &QDialogButtonBox::accepted, this, &OpenRouteDialog::loadRoute);
  QObject::connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(route_list->list, &QListWidget::itemDoubleClicked, [this](QListWidgetItem *item) {
    route_edit->setText(item->text());
  });
  QObject::connect(route_edit, &QLineEdit::textChanged, [this]() {
    btn_box->button(QDialogButtonBox::Open)->setEnabled(!route_edit->text().isEmpty());
  });
  QObject::connect(file_btn, &QPushButton::clicked, [this]() {
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

RemoteRouteList::RemoteRouteList(QWidget *parent) : QStackedWidget(parent) {
  QWidget *w = new QWidget(this);
  QVBoxLayout *main_layout = new QVBoxLayout(w);

  QHBoxLayout *dongleid_layout = new QHBoxLayout;
  dongleid_layout->addWidget(new QLabel(tr("dongle id:")));
  dongleid_layout->addWidget(dongleid_cb = new QComboBox);
  main_layout->addLayout(dongleid_layout);
  main_layout->addWidget(list = new QListWidget);

  QWidget *loading_w = new QWidget(this);
  QVBoxLayout *v = new QVBoxLayout(loading_w);
  msg_label = new QLabel(tr("Loading routes from server..."));
  msg_label->setAlignment(Qt::AlignCenter);
  v->addStretch(0);
  v->addWidget(msg_label);
  QHBoxLayout *h = new QHBoxLayout();
  h->addStretch(0);
  h->addWidget(retry_btn = new QPushButton(tr("Retry")));
  h->addStretch(0);
  v->addLayout(h);
  v->addStretch(0);

  addWidget(loading_w);
  addWidget(w);

  QObject::connect(dongleid_cb, &QComboBox::currentTextChanged, this, &RemoteRouteList::getRouteList);
  QObject::connect(retry_btn, &QPushButton::clicked, this, &RemoteRouteList::getDevices);

  getDevices();
}

void RemoteRouteList::getDevices() {
  msg_label->setText(tr("Loading routes from server..."));
  retry_btn->setVisible(false);
  dongleid_cb->clear();
  HttpRequest *http = new HttpRequest(this, false);
  QObject::connect(http, &HttpRequest::requestDone, [=](const QString &json, bool success, QNetworkReply::NetworkError error) {
    if (success) {
      auto doc = QJsonDocument::fromJson(json.trimmed().toUtf8());
      if (!doc.isEmpty() && doc.isArray()) {
        for (const auto &device : QJsonDocument::fromJson(json.trimmed().toUtf8()).array()) {
          dongleid_cb->addItem(device.toObject()["dongle_id"].toString());
        }
      }
    }
    if (dongleid_cb->count() == 0) {
      QString text = tr("Failed to load routes from server\n");
      if (error == QNetworkReply::ContentAccessDenied || error == QNetworkReply::AuthenticationRequiredError) {
        text += tr("Unauthorized. Authenticate with tools/lib/auth.py");
      }
      msg_label->setText(text);
      retry_btn->setVisible(true);
    }
    http->deleteLater();
  });
  http->sendRequest("https://api.commadotai.com/v1/me/devices/");
}

void RemoteRouteList::getRouteList(const QString &dongleid) {
  list->clear();
  if (!dongleid.isEmpty()) {
    HttpRequest *http = new HttpRequest(this, false);
    QObject::connect(http, &HttpRequest::requestDone, [=](const QString &json, bool success, QNetworkReply::NetworkError error) {
      if (success) {
        auto doc = QJsonDocument::fromJson(json.trimmed().toUtf8());
        if (!doc.isEmpty() && doc.isArray()) {
          for (const auto &route : doc.array()) {
            list->addItem(route.toObject()["canonical_route_name"].toString());
          }
        }
        setCurrentIndex(1);
      } else {
        msg_label->setText(tr("Failed to load routes from server\n"));
        retry_btn->setVisible(true);
      }
      http->deleteLater();
    });
    http->sendRequest(QString("https://api.commadotai.com/v1/%1/segments").arg(dongleid));
  }
}
