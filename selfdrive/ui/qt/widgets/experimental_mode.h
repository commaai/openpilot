#pragma once

#include <QLabel>
#include <QPushButton>

class ExperimentalModeButton : public QPushButton {
  Q_OBJECT
  Q_PROPERTY(bool experimental_mode MEMBER experimental_mode);

public:
  explicit ExperimentalModeButton(QWidget* parent = 0);

signals:
  void openSettings(int index = 0);

private:
  void showEvent(QShowEvent *event) override;

  bool experimental_mode;
  QPixmap experimental_pixmap;
  QPixmap chill_pixmap;
  QLabel *mode_label;
  QLabel *mode_icon;
};
