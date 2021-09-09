#pragma once

#include <QStackedLayout>
#include <QWidget>

#include "selfdrive/ui/qt/widgets/cameraview.h"
#include "selfdrive/ui/ui.h"


// ***** onroad widgets *****

class OnroadHud : public QWidget {
  Q_OBJECT
  Q_PROPERTY(QString speed MEMBER speed_ NOTIFY valueChanged);
  Q_PROPERTY(QString speedUnit MEMBER speedUnit_ NOTIFY valueChanged);
  Q_PROPERTY(QString maxSpeed MEMBER maxSpeed_ NOTIFY valueChanged);
  Q_PROPERTY(bool engageable MEMBER engageable_ NOTIFY valueChanged);
  Q_PROPERTY(bool dmActive MEMBER dmActive_ NOTIFY valueChanged);
  Q_PROPERTY(bool hideDM MEMBER hideDM_ NOTIFY valueChanged);
  Q_PROPERTY(int status MEMBER status_ NOTIFY valueChanged);

public:
  OnroadHud(QWidget *parent = 0);
  void updateAlert(const Alert &a, const QColor &color);
public slots:
  void updateState(const UIState &s);

protected:
  void paintEvent(QPaintEvent*) override;
  void drawIcon(QPainter &p, int x, int y, QPixmap &img, QBrush bg, float opacity);
  void drawText(QPainter &p, int x, int y, Qt::Alignment flag, const QString &text, int alpha = 255);
  void drawAlert(QPainter &p);

private:
  QColor bg;
  Alert alert = {};
  QPixmap engage_img, dm_img;
  const int radius = 192;
  const int img_size = 135;
  QString speed_, speedUnit_, maxSpeed_;
  bool metric = false, engageable_ = false, dmActive_ = false, hideDM_ = false;
  int status_ = STATUS_DISENGAGED;

signals:
  void valueChanged();
};

// container window for the NVG UI
class NvgWindow : public CameraViewWidget {
  Q_OBJECT

public:
  explicit NvgWindow(VisionStreamType type, QWidget* parent = 0) : CameraViewWidget(type, true, parent) {}

protected:
  void paintGL() override;
  void initializeGL() override;
  double prev_draw_t = 0;
};

// container for all onroad widgets
class OnroadWindow : public QWidget {
  Q_OBJECT

public:
  OnroadWindow(QWidget* parent = 0);
  bool isMapVisible() const { return map && map->isVisible(); }

private:
  void paintEvent(QPaintEvent *event);
  void mousePressEvent(QMouseEvent* e) override;
  OnroadHud *hud;
  NvgWindow *nvg;
  QColor bg = bg_colors[STATUS_DISENGAGED];
  QWidget *map = nullptr;
  QHBoxLayout* split;

signals:
  void updateStateSignal(const UIState &s);
  void offroadTransitionSignal(bool offroad);

private slots:
  void offroadTransition(bool offroad);
  void updateState(const UIState &s);
};
