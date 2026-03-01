#pragma once

#include <QObject>
#include <QString>

class PyApiRequest : public QObject {
  Q_OBJECT

public:
  explicit PyApiRequest(QObject *parent = nullptr) : QObject(parent) {}

  void fetchDevices();
  void fetchRoutes(const QString &dongle_id, int64_t start_ms = 0, int64_t end_ms = 0, bool preserved = false);

  // Cancel pending request (best-effort, will just discard results)
  void cancel() { cancelled_ = true; }

signals:
  void requestDone(const QString &response, bool success, int error_code);

private:
  bool cancelled_ = false;
};
