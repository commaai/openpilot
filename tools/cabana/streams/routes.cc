#include "tools/cabana/streams/routes.h"

#include <QDateTime>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>

class OneShotHttpRequest : public HttpRequest {
public:
  OneShotHttpRequest(QObject *parent) : HttpRequest(parent, false) {}
  void send(const QString &url) {
    if (reply) {
      reply->disconnect();
      reply->abort();
      reply->deleteLater();
      reply = nullptr;
    }
    sendRequest(url);
  }
};

// The RouteListWidget class extends QListWidget to display a custom message when empty
class RouteListWidget : public QListWidget {
public:
  RouteListWidget(QWidget *parent = nullptr) : QListWidget(parent) {}
  void setEmptyText(const QString &text) {
    empty_text_ = text;
    viewport()->update();
  }
  void paintEvent(QPaintEvent *event) override {
    QListWidget::paintEvent(event);
    if (count() == 0) {
      QPainter painter(viewport());
      painter.drawText(viewport()->rect(), Qt::AlignCenter, empty_text_);
    }
  }
  QString empty_text_ = tr("No items");
};

RoutesDialog::RoutesDialog(QWidget *parent) : QDialog(parent), route_requester_(new OneShotHttpRequest(this)) {
  setWindowTitle(tr("Remote routes"));

  QFormLayout *layout = new QFormLayout(this);
  layout->addRow(tr("Device"), device_list_ = new QComboBox(this));
  layout->addRow(period_selector_ = new QComboBox(this));
  layout->addRow(route_list_ = new RouteListWidget(this));
  auto button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  layout->addRow(button_box);

  device_list_->addItem(tr("Loading..."));
  // Populate period selector with predefined durations
  period_selector_->addItem(tr("Last week"), 7);
  period_selector_->addItem(tr("Last 2 weeks"), 14);
  period_selector_->addItem(tr("Last month"), 30);
  period_selector_->addItem(tr("Last 6 months"), 180);
  period_selector_->addItem(tr("Preserved"), -1);

  // Connect signals and slots
  QObject::connect(route_requester_, &HttpRequest::requestDone, this, &RoutesDialog::parseRouteList);
  connect(device_list_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RoutesDialog::fetchRoutes);
  connect(period_selector_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RoutesDialog::fetchRoutes);
  connect(route_list_, &QListWidget::itemDoubleClicked, this, &QDialog::accept);
  QObject::connect(button_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
  QObject::connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

  // Send request to fetch devices
  HttpRequest *http = new HttpRequest(this, false);
  QObject::connect(http, &HttpRequest::requestDone, this, &RoutesDialog::parseDeviceList);
  http->sendRequest(CommaApi::BASE_URL + "/v1/me/devices/");
}

void RoutesDialog::parseDeviceList(const QString &json, bool success, QNetworkReply::NetworkError err) {
  if (success) {
    device_list_->clear();
    auto devices = QJsonDocument::fromJson(json.toUtf8()).array();
    for (const QJsonValue &device : devices) {
      QString dongle_id = device["dongle_id"].toString();
      device_list_->addItem(dongle_id, dongle_id);
    }
  } else {
    bool unauthorized = (err == QNetworkReply::ContentAccessDenied || err == QNetworkReply::AuthenticationRequiredError);
    QMessageBox::warning(this, tr("Error"), unauthorized ? tr("Unauthorized, Authenticate with tools/lib/auth.py") : tr("Network error"));
    reject();
  }
  sender()->deleteLater();
}

void RoutesDialog::fetchRoutes() {
  if (device_list_->currentIndex() == -1 || device_list_->currentData().isNull())
    return;

  route_list_->clear();
  route_list_->setEmptyText(tr("Loading..."));
  // Construct URL with selected device and date range
  QString url = QString("%1/v1/devices/%2").arg(CommaApi::BASE_URL, device_list_->currentText());
  int period = period_selector_->currentData().toInt();
  if (period == -1) {
    url += "/routes/preserved";
  } else {
    QDateTime now = QDateTime::currentDateTime();
    url += QString("/routes_segments?start=%1&end=%2")
               .arg(now.addDays(-period).toMSecsSinceEpoch())
               .arg(now.toMSecsSinceEpoch());
  }
  route_requester_->send(url);
}

void RoutesDialog::parseRouteList(const QString &json, bool success, QNetworkReply::NetworkError err) {
  if (success) {
    for (const QJsonValue &route : QJsonDocument::fromJson(json.toUtf8()).array()) {
      QDateTime from, to;
      if (period_selector_->currentData().toInt() == -1) {
        from = QDateTime::fromString(route["start_time"].toString(), Qt::ISODateWithMs);
        to = QDateTime::fromString(route["end_time"].toString(), Qt::ISODateWithMs);
      } else {
        from = QDateTime::fromMSecsSinceEpoch(route["start_time_utc_millis"].toDouble());
        to = QDateTime::fromMSecsSinceEpoch(route["end_time_utc_millis"].toDouble());
      }
      auto item = new QListWidgetItem(QString("%1    %2min").arg(from.toString()).arg(from.secsTo(to) / 60));
      item->setData(Qt::UserRole, route["fullname"].toString());
      route_list_->addItem(item);
    }
    if (route_list_->count() > 0) route_list_->setCurrentRow(0);
  } else {
    QMessageBox::warning(this, tr("Error"), tr("Failed to fetch routes. Check your network connection."));
    reject();
  }
  route_list_->setEmptyText(tr("No items"));
}

QString RoutesDialog::route() {
  auto current_item = route_list_->currentItem();
  return current_item ? current_item->data(Qt::UserRole).toString() : "";
}
