#pragma once
#include <QWidget>
#include <QLabel>

class MapPanel : public QWidget {
  Q_OBJECT
public:
  explicit MapPanel(QWidget* parent = nullptr);

  void parseResponse(const QString& response);

private:
  QLabel *home_address, *home_icon;
  QLabel *work_address, *work_icon;
};
