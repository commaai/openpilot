#include "tools/cabana/streams/routes.h"

#include <QDateTime>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>

#include "system/hardware/hw.h"

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

RoutesDialog::RoutesDialog(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Remote routes"));

  QFormLayout *layout = new QFormLayout(this);
  layout->addRow(tr("Device"), device_list_ = new QComboBox(this));
  layout->addRow(tr("Duration"), period_selector_ = new QComboBox(this));
  layout->addRow(route_list_ = new RouteListWidget(this));
  auto button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  layout->addRow(button_box);

  device_list_->addItem(tr("Loading..."));
  // Populate period selector with predefined durations
  period_selector_->addItem(tr("Last week"), 7);
  period_selector_->addItem(tr("Last 2 weeks"), 14);
  period_selector_->addItem(tr("Last month"), 30);
  period_selector_->addItem(tr("Last 6 months"), 180);

  // Connect signals and slots
  connect(device_list_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RoutesDialog::fetchRoutes);
  connect(period_selector_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RoutesDialog::fetchRoutes);
  connect(route_list_, &QListWidget::itemDoubleClicked, this, &QDialog::accept);
  QObject::connect(button_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
  QObject::connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

  // Send request to fetch devices
  HttpRequest *http = new HttpRequest(this, !Hardware::PC());
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

  HttpRequest *http = new HttpRequest(this, !Hardware::PC());
  QObject::connect(http, &HttpRequest::requestDone, this, &RoutesDialog::parseRouteList);

  // Construct URL with selected device and date range
  auto dongle_id = device_list_->currentData().toString();
  QDateTime current = QDateTime::currentDateTime();
  QString url = QString("%1/v1/devices/%2/routes_segments?start=%3&end=%4")
                    .arg(CommaApi::BASE_URL).arg(dongle_id)
                    .arg(current.addDays(-(period_selector_->currentData().toInt())).toMSecsSinceEpoch())
                    .arg(current.toMSecsSinceEpoch());
  http->sendRequest(url);
}

void RoutesDialog::parseRouteList(const QString &json, bool success, QNetworkReply::NetworkError err) {
  if (success) {
    for (const QJsonValue &route : QJsonDocument::fromJson(json.toUtf8()).array()) {
      uint64_t start_time = route["start_time_utc_millis"].toDouble();
      uint64_t end_time = route["end_time_utc_millis"].toDouble();
      auto datetime = QDateTime::fromMSecsSinceEpoch(start_time);
      auto item = new QListWidgetItem(QString("%1    %2min").arg(datetime.toString()).arg((end_time - start_time) / (1000 * 60)));
      item->setData(Qt::UserRole, route["fullname"].toString());
      route_list_->addItem(item);
    }
    // Select first route if available
    if (route_list_->count() > 0) route_list_->setCurrentRow(0);
  } else {
    QMessageBox::warning(this, tr("Error"), tr("Failed to fetch routes. Check your network connection."));
    reject();
  }
  route_list_->setEmptyText(tr("No items"));
  sender()->deleteLater();
}

void RoutesDialog::accept() {
  if (auto current_item = route_list_->currentItem()) {
    route_ = current_item->data(Qt::UserRole).toString();
  }
  QDialog::accept();
}
