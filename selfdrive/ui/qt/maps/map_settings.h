#pragma once
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <QStackedWidget>

#include "selfdrive/common/params.h"

class MapPanel : public QWidget {
  Q_OBJECT
public:
  explicit MapPanel(QWidget* parent = nullptr);

  void navigateTo(const QJsonObject &place);
  void parseResponse(const QString &response);
  void failedResponse(const QString &response);
  void clear();

private:
  Params params;
  QStackedWidget *stack;
  QPushButton *home_button, *work_button;
  QLabel *home_address, *work_address;
  QVBoxLayout *recent_layout;

signals:
  void closeSettings();
};
