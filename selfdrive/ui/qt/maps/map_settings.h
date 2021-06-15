#pragma once
#include <QWidget>
#include <QLabel>
#include <QPushButton>

class MapPanel : public QWidget {
  Q_OBJECT
public:
  explicit MapPanel(QWidget* parent = nullptr);

  void parseResponse(const QString& response);

private:
  QPushButton *home_button, *work_button;
  QLabel *home_address, *work_address;
};
