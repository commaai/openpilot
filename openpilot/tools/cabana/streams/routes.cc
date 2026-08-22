#include "tools/cabana/streams/routes.h"

#include <chrono>
#include <ctime>
#include <string>
#include <thread>
#include <utility>

#include <QApplication>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>

#include "json11/json11.hpp"
#include "tools/replay/py_downloader.h"

namespace {

// Parse a PyDownloader JSON response into (success, error_code).
std::pair<bool, int> checkApiResponse(const std::string &result) {
  if (result.empty()) return {false, 500};
  std::string err;
  auto doc = json11::Json::parse(result, err);
  if (!err.empty()) return {false, 500};
  if (doc.is_object() && doc["error"].is_string()) {
    return {false, doc["error"].string_value() == "unauthorized" ? 401 : 500};
  }
  return {true, 0};
}

int64_t nowUnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Parse ISO-8601 (with optional fractional seconds / Z) to unix ms. Returns 0 on failure.
int64_t parseIsoToUnixMs(const std::string &s) {
  std::string bytes = s;
  if (!bytes.empty() && (bytes.back() == 'Z' || bytes.back() == 'z')) bytes.pop_back();
  int millis = 0;
  auto dot = bytes.find('.');
  if (dot != std::string::npos) {
    std::string frac = bytes.substr(dot + 1);
    bytes = bytes.substr(0, dot);
    while (frac.size() < 3) frac.push_back('0');
    millis = std::atoi(frac.substr(0, 3).c_str());
  }
  std::tm tm{};
  const char *ret = strptime(bytes.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
  if (!ret) ret = strptime(bytes.c_str(), "%Y-%m-%d %H:%M:%S", &tm);
  if (!ret) return 0;
  tm.tm_isdst = -1;
  time_t secs = timegm(&tm);
  if (secs == static_cast<time_t>(-1)) return 0;
  return static_cast<int64_t>(secs) * 1000 + millis;
}

QString formatUnixMs(int64_t ms) {
  time_t secs = static_cast<time_t>(ms / 1000);
  std::tm tm{};
  localtime_r(&secs, &tm);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
  return QString::fromUtf8(buf);
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
  std::thread([this, alive = std::weak_ptr<bool>(alive_)]() {
    std::string result = PyDownloader::getDevices();
    QMetaObject::invokeMethod(qApp, [this, alive, r = QString::fromStdString(result), response = checkApiResponse(result)]() {
      if (!alive.expired()) parseDeviceList(r, response.first, response.second);
    }, Qt::QueuedConnection);
  }).detach();
}

void RoutesDialog::parseDeviceList(const QString &json, bool success, int error_code) {
  if (success) {
    device_list_->clear();
    std::string err;
    auto doc = json11::Json::parse(json.toStdString(), err);
    if (err.empty() && doc.is_array()) {
      for (const auto &device : doc.array_items()) {
        QString dongle_id = QString::fromStdString(device["dongle_id"].string_value());
        device_list_->addItem(dongle_id, dongle_id);
      }
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

  std::string did = device_list_->currentText().toStdString();
  int period = period_selector_->currentData().toInt();

  bool preserved = (period == -1);
  int64_t start_ms = 0, end_ms = 0;
  if (!preserved) {
    end_ms = nowUnixMs();
    start_ms = end_ms - static_cast<int64_t>(period) * 24LL * 60LL * 60LL * 1000LL;
  }

  int request_id = ++fetch_id_;
  std::thread([this, alive = std::weak_ptr<bool>(alive_), did, start_ms, end_ms, preserved, request_id]() {
    std::string result = PyDownloader::getDeviceRoutes(did, start_ms, end_ms, preserved);
    QMetaObject::invokeMethod(qApp, [this, alive, r = QString::fromStdString(result), response = checkApiResponse(result), request_id]() {
      if (!alive.expired() && fetch_id_ == request_id) parseRouteList(r, response.first, response.second);
    }, Qt::QueuedConnection);
  }).detach();
}

void RoutesDialog::parseRouteList(const QString &json, bool success, int error_code) {
  if (success) {
    std::string err;
    auto doc = json11::Json::parse(json.toStdString(), err);
    if (err.empty() && doc.is_array()) {
      for (const auto &route : doc.array_items()) {
        int64_t from_ms = 0, to_ms = 0;
        if (period_selector_->currentData().toInt() == -1) {
          from_ms = parseIsoToUnixMs(route["start_time"].string_value());
          to_ms = parseIsoToUnixMs(route["end_time"].string_value());
        } else {
          from_ms = static_cast<int64_t>(route["start_time_utc_millis"].number_value());
          to_ms = static_cast<int64_t>(route["end_time_utc_millis"].number_value());
        }
        const int mins = static_cast<int>((to_ms - from_ms) / 60000);
        auto item = new QListWidgetItem(QString("%1    %2min").arg(formatUnixMs(from_ms)).arg(mins));
        item->setData(Qt::UserRole, QString::fromStdString(route["fullname"].string_value()));
        route_list_->addItem(item);
      }
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
