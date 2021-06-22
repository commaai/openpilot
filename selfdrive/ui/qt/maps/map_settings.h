#pragma once
#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>

class MapPanel : public QWidget {
  Q_OBJECT
public:
  explicit MapPanel(QWidget* parent = nullptr);

  void navigateTo(const QJsonObject &place);
  void parseResponse(const QString &response);
  void clear();

private:
  QPushButton *home_button, *work_button;
  QLabel *home_address, *work_address;

signals:
  void closeSettings();
};
