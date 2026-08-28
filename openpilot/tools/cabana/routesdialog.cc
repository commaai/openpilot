#include "tools/cabana/routesdialog.h"

#include <string>
#include <utility>

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>

#include "tools/cabana/utils/util.h"

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

  routes::fetchDevices([this, alive = std::weak_ptr<bool>(alive_)](std::vector<routes::DeviceInfo> devices, bool success, int error_code) {
    utils::runOnMainThread([this, alive, devices = std::move(devices), success, error_code]() {
      if (!alive.expired()) setDeviceList(devices, success, error_code);
    });
  });
}

void RoutesDialog::setDeviceList(const std::vector<routes::DeviceInfo> &devices, bool success, int error_code) {
  if (success) {
    device_list_->clear();
    for (const auto &device : devices) {
      QString dongle_id = QString::fromStdString(device.dongle_id);
      device_list_->addItem(dongle_id, dongle_id);
    }
  } else {
    QMessageBox::warning(this, tr("Error"), error_code == 401 ? tr("Unauthorized. Authenticate with openpilot/tools/lib/auth.py") : tr("Network error"));
    reject();
  }
}

void RoutesDialog::fetchRoutes() {
  if (device_list_->currentIndex() == -1 || device_list_->currentData().isNull())
    return;

  route_list_->clear();
  route_list_->setEmptyText(tr("Loading..."));

  int request_id = ++fetch_id_;
  auto on_routes = [this, alive = std::weak_ptr<bool>(alive_), request_id](std::vector<routes::RouteInfo> list, bool success, int) {
    utils::runOnMainThread([this, alive, list = std::move(list), success, request_id]() {
      if (!alive.expired() && fetch_id_ == request_id) setRouteList(list, success);
    });
  };
  routes::fetchRoutes(device_list_->currentText().toStdString(), period_selector_->currentData().toInt(), std::move(on_routes));
}

void RoutesDialog::setRouteList(const std::vector<routes::RouteInfo> &list, bool success) {
  if (success) {
    for (const auto &route : list) {
      const int mins = static_cast<int>((route.end_ms - route.start_ms) / 60000);
      auto item = new QListWidgetItem(QString::fromStdString(routes::formatUnixMs(route.start_ms) + "    " + std::to_string(mins) + "min"));
      item->setData(Qt::UserRole, QString::fromStdString(route.name));
      route_list_->addItem(item);
    }
    if (route_list_->count() > 0) route_list_->setCurrentRow(0);
  } else {
    QMessageBox::warning(this, tr("Error"), tr("Failed to fetch routes. Check your network connection."));
    reject();
  }
  route_list_->setEmptyText(tr("No items"));
}

std::string RoutesDialog::route() {
  auto current_item = route_list_->currentItem();
  return current_item ? current_item->data(Qt::UserRole).toString().toStdString() : "";
}
