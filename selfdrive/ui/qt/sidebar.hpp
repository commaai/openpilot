#pragma once

#include <QtWidgets>


class SettingsBtn : public QAbstractButton {
  Q_OBJECT

public:
  SettingsBtn(QWidget* parent = 0);
protected:
  void paintEvent(QPaintEvent*) override;
private:
  QImage image;
};

class StatusWidget : public QFrame {
  Q_OBJECT

public:
  StatusWidget(QString text, QWidget* parent = 0);
  QLabel* label;

protected:
  void paintEvent(QPaintEvent*) override;
};

class Sidebar : public QFrame {
  Q_OBJECT


public:
  explicit Sidebar(QWidget* parent = 0);
};