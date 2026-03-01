#include "tools/cabana/streams/routes.h"

#include <QApplication>
#include <QDateTime>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>
#include <QPointer>
#include <thread>

#include "tools/replay/py_downloader.h"

namespace {

// Parse a PyDownloader JSON response into (success, error_code).
std::pair<bool, int> checkApiResponse(const std::string &result) {
  if (result.empty()) return {false, 500};
  auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(result));
  if (doc.isObject() && doc.object().contains("error")) {
    return {false, doc.object()["error"].toString() == "unauthorized" ? 401 : 500};
  }
  return {true, 0};
}

}  // namespace

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
  layout->addRow(period_selector_ = new QComboBox(this));
  layout->addRow(route_list_ = new RouteListWidget(this));
  auto button_box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  layout->addRow(button_box);

  device_list_->addItem(tr("Loading..."));
  period_selector_->addItem(tr("Last week"), 7);
  period_selector_->addItem(tr("Last 2 weeks"), 14);
  period_selector_->addItem(tr("Last month"), 30);
  period_selector_->addItem(tr("Last 6 months"), 180);
  period_selector_->addItem(tr("Preserved"), -1);

  connect(device_list_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RoutesDialog::fetchRoutes);
  connect(period_selector_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RoutesDialog::fetchRoutes);
  connect(route_list_, &QListWidget::itemDoubleClicked, this, &QDialog::accept);
  connect(button_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

  // Fetch devices
  QPointer<RoutesDialog> self = this;
  std::thread([self]() {
    std::string result = PyDownloader::getDevices();
    auto [success, error_code] = checkApiResponse(result);
    QMetaObject::invokeMethod(qApp, [self, r = QString::fromStdString(result), success, error_code]() {
      if (self) self->parseDeviceList(r, success, error_code);
    }, Qt::QueuedConnection);
  }).detach();
}

void RoutesDialog::parseDeviceList(const QString &json, bool success, int error_code) {
  if (success) {
    device_list_->clear();
    for (const QJsonValue &device : QJsonDocument::fromJson(json.toUtf8()).array()) {
      QString dongle_id = device["dongle_id"].toString();
      device_list_->addItem(dongle_id, dongle_id);
    }
  } else {
    QMessageBox::warning(this, tr("Error"), error_code == 401 ? tr("Unauthorized. Authenticate with tools/lib/auth.py") : tr("Network error"));
    reject();
  }
}

void RoutesDialog::fetchRoutes() {
  if (device_list_->currentIndex() == -1 || device_list_->currentData().isNull())
    return;

  route_list_->clear();
  route_list_->setEmptyText(tr("Loading..."));

  std::string did = device_list_->currentText().toStdString();
  int period = period_selector_->currentData().toInt();

  bool preserved = (period == -1);
  int64_t start_ms = 0, end_ms = 0;
  if (!preserved) {
    QDateTime now = QDateTime::currentDateTime();
    start_ms = now.addDays(-period).toMSecsSinceEpoch();
    end_ms = now.toMSecsSinceEpoch();
  }

  int request_id = ++fetch_id_;
  QPointer<RoutesDialog> self = this;
  std::thread([self, did, start_ms, end_ms, preserved, request_id]() {
    std::string result = PyDownloader::getDeviceRoutes(did, start_ms, end_ms, preserved);
    if (!self || self->fetch_id_ != request_id) return;
    auto [success, error_code] = checkApiResponse(result);
    QMetaObject::invokeMethod(qApp, [self, r = QString::fromStdString(result), success, error_code, request_id]() {
      if (self && self->fetch_id_ == request_id) self->parseRouteList(r, success, error_code);
    }, Qt::QueuedConnection);
  }).detach();
}

void RoutesDialog::parseRouteList(const QString &json, bool success, int error_code) {
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
