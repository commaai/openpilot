#include "tools/cabana/streams/routes.h"

#include <QDateTime>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>

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

RoutesDialog::RoutesDialog(QWidget *parent)
    : QDialog(parent),
      device_requester_(new PyApiRequest(this)),
      route_requester_(new PyApiRequest(this)) {
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
  QObject::connect(route_requester_, &PyApiRequest::requestDone, this, &RoutesDialog::parseRouteList);
  connect(device_list_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RoutesDialog::fetchRoutes);
  connect(period_selector_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RoutesDialog::fetchRoutes);
  connect(route_list_, &QListWidget::itemDoubleClicked, this, &QDialog::accept);
  QObject::connect(button_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
  QObject::connect(button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);

  // Fetch devices
  QObject::connect(device_requester_, &PyApiRequest::requestDone, this, &RoutesDialog::parseDeviceList);
  device_requester_->fetchDevices();
}

void RoutesDialog::parseDeviceList(const QString &json, bool success, int error_code) {
  if (success) {
    device_list_->clear();
    auto devices = QJsonDocument::fromJson(json.toUtf8()).array();
    for (const QJsonValue &device : devices) {
      QString dongle_id = device["dongle_id"].toString();
      device_list_->addItem(dongle_id, dongle_id);
    }
  } else {
    bool unauthorized = (error_code == 401);
    QMessageBox::warning(this, tr("Error"), unauthorized ? tr("Unauthorized, Authenticate with tools/lib/auth.py") : tr("Network error"));
    reject();
  }
}

void RoutesDialog::fetchRoutes() {
  if (device_list_->currentIndex() == -1 || device_list_->currentData().isNull())
    return;

  route_list_->clear();
  route_list_->setEmptyText(tr("Loading..."));

  QString dongle_id = device_list_->currentText();
  int period = period_selector_->currentData().toInt();

  route_requester_->cancel();
  if (period == -1) {
    route_requester_->fetchRoutes(dongle_id, 0, 0, true);
  } else {
    QDateTime now = QDateTime::currentDateTime();
    int64_t start_ms = now.addDays(-period).toMSecsSinceEpoch();
    int64_t end_ms = now.toMSecsSinceEpoch();
    route_requester_->fetchRoutes(dongle_id, start_ms, end_ms, false);
  }
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
