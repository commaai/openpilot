#include "tools/cabana/utils/pyapi.h"

#include <QtConcurrent>
#include <QJsonDocument>
#include <QJsonObject>

#include "tools/replay/py_downloader.h"

void PyApiRequest::fetchDevices() {
  cancelled_ = false;
  QtConcurrent::run([this]() {
    std::string result = PyDownloader::getDevices();
    if (cancelled_) return;

    bool success = true;
    int error_code = 0;

    if (!result.empty()) {
      auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(result));
      if (doc.isObject() && doc.object().contains("error")) {
        success = false;
        QString err = doc.object()["error"].toString();
        error_code = (err == "unauthorized") ? 401 : 500;
      }
    } else {
      success = false;
      error_code = 500;
    }

    emit requestDone(QString::fromStdString(result), success, error_code);
  });
}

void PyApiRequest::fetchRoutes(const QString &dongle_id, int64_t start_ms, int64_t end_ms, bool preserved) {
  cancelled_ = false;
  std::string did = dongle_id.toStdString();
  QtConcurrent::run([this, did, start_ms, end_ms, preserved]() {
    std::string result = PyDownloader::getDeviceRoutes(did, start_ms, end_ms, preserved);
    if (cancelled_) return;

    bool success = true;
    int error_code = 0;

    if (!result.empty()) {
      auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(result));
      if (doc.isObject() && doc.object().contains("error")) {
        success = false;
        QString err = doc.object()["error"].toString();
        error_code = (err == "unauthorized") ? 401 : 500;
      }
    } else {
      success = false;
      error_code = 500;
    }

    emit requestDone(QString::fromStdString(result), success, error_code);
  });
}
