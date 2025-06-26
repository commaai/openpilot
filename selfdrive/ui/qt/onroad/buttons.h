#pragma once

#include <QPushButton>

#include "selfdrive/ui/ui.h"

const int btn_size = 192;
const int img_size = (btn_size / 4) * 3;

class ExperimentalButton : public QPushButton {
  Q_OBJECT

public:
  explicit ExperimentalButton(QWidget *parent = 0);
  void updateState(const UIState &s);

private:
  void paintEvent(QPaintEvent *event) override;
  void changeMode();

  Params params;
  QPixmap engage_img;
  QPixmap experimental_img;
  bool experimental_mode;
  bool engageable;
};

class RecordingAudioButton : public QPushButton {
  Q_OBJECT

public:
  explicit RecordingAudioButton(QWidget *parent = 0);
  void updateState(const UIState &s);

signals:
  void openSettings(int index = 0, const QString &param = "");

private:
  void paintEvent(QPaintEvent *event) override;

  QPixmap microphone_img;
  bool recording_audio;
};

void drawIcon(QPainter &p, const QPoint &center, const QPixmap &img, const QBrush &bg, float opacity);
