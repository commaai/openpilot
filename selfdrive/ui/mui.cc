#include <QApplication>
#include <QtWidgets>
#include <QTimer>
#include <QGraphicsScene>

#include "cereal/messaging/messaging.h"
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/qt_window.h"

class StatusBar : public QGraphicsRectItem {
  private:
    QLinearGradient linear_gradient;
    QRadialGradient radial_gradient;
    QTimer animation_timer;
    const int animation_length = 10;
    int animation_index = 0;

  public:
    StatusBar(double x, double y, double width, double height) : QGraphicsRectItem {x, y, width, height} {
      linear_gradient = QLinearGradient(0, 0, 0, height/2);
      linear_gradient.setSpread(QGradient::ReflectSpread);

      radial_gradient = QRadialGradient(width/2, height/2, width/8);
      QObject::connect(&animation_timer, &QTimer::timeout, [=]() {
        animation_index++;
        animation_index %= animation_length;
      });
      animation_timer.start(50);
    }

    void solidColor(QColor color) {
      QColor dark_color = QColor(color);
      dark_color.setAlphaF(0.5);

      linear_gradient.setColorAt(0, dark_color);
      linear_gradient.setColorAt(1, color);
      setBrush(QBrush(linear_gradient));
    }

    // these need to be called continuously for the animations to work.
    // can probably clean that up with some more abstractions
    void blinkingColor(QColor color) {
      QColor dark_color = QColor(color);
      dark_color.setAlphaF(0.1);

      int radius = (rect().width() / animation_length) * animation_index;
      QPoint center = QPoint(rect().width()/2, rect().height()/2);
      radial_gradient.setCenter(center);
      radial_gradient.setFocalPoint(center);
      radial_gradient.setRadius(radius);

      radial_gradient.setColorAt(1, dark_color);
      radial_gradient.setColorAt(0, color);
      setBrush(QBrush(radial_gradient));
    }

    void laneChange(cereal::LateralPlan::LaneChangeDirection direction) {
      QColor dark_color = QColor(bg_colors[STATUS_ENGAGED]);
      dark_color.setAlphaF(0.1);

      int x = (rect().width() / animation_length) * animation_index;
      QPoint center = QPoint(((direction == cereal::LateralPlan::LaneChangeDirection::RIGHT) ? x : (rect().width() - x)), rect().height()/2);
      radial_gradient.setCenter(center);
      radial_gradient.setFocalPoint(center);
      radial_gradient.setRadius(rect().width()/5);

      radial_gradient.setColorAt(1, dark_color);
      radial_gradient.setColorAt(0, bg_colors[STATUS_ENGAGED]);
      setBrush(QBrush(radial_gradient));
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override {
      painter->setPen(QPen());
      painter->setBrush(brush());

      double rounding_radius = rect().height()/2;
      painter->drawRoundedRect(rect(), rounding_radius, rounding_radius);
    }
};

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget w;
  setMainWindow(&w);

  w.setStyleSheet("background-color: black;");

  // our beautiful UI
  QVBoxLayout *layout = new QVBoxLayout(&w);

  QGraphicsScene *scene = new QGraphicsScene();
  StatusBar *status_bar = new StatusBar(0, 0, 1000, 50);
  scene->addItem(status_bar);

  QGraphicsView *graphics_view = new QGraphicsView(scene);
  layout->insertSpacing(0, 400);
  layout->addWidget(graphics_view, 0, Qt::AlignCenter);

  QTimer timer;
  QObject::connect(&timer, &QTimer::timeout, [=]() {
    static SubMaster sm({"deviceState", "controlsState", "lateralPlan"});

    bool onroad_prev = sm.allAliveAndValid({"deviceState"}) &&
                       sm["deviceState"].getDeviceState().getStarted();
    sm.update(0);

    bool onroad = sm.allAliveAndValid({"deviceState"}) &&
                  sm["deviceState"].getDeviceState().getStarted();

    if (onroad) {
      auto cs = sm["controlsState"].getControlsState();
      UIStatus status = cs.getEnabled() ? STATUS_ENGAGED : STATUS_DISENGAGED;
      auto lp = sm["lateralPlan"].getLateralPlan();
      if (lp.getLaneChangeState() == cereal::LateralPlan::LaneChangeState::PRE_LANE_CHANGE) {
        status_bar->blinkingColor(bg_colors[status]);
      } else if (lp.getLaneChangeState() == cereal::LateralPlan::LaneChangeState::LANE_CHANGE_STARTING ||
                 lp.getLaneChangeState() == cereal::LateralPlan::LaneChangeState::LANE_CHANGE_FINISHING) {
        status_bar->laneChange(lp.getLaneChangeDirection());
      } else {
        status_bar->solidColor(bg_colors[status]);
      }
    }

    if ((onroad != onroad_prev) || sm.frame < 2) {
      Hardware::set_brightness(50);
      Hardware::set_display_power(onroad);
    }
  });
  timer.start(50);

  return a.exec();
}
