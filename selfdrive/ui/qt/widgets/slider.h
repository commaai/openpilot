#pragma once
#include <QTimer>
#include <QLabel>
#include <QSlider>
#include <QWidget>
#include <QMouseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>

#include <functional>
#include "common/params.h"
#include "selfdrive/ui/ui.h"

class CustomSlider : public QSlider {
  Q_OBJECT

public:
  using CerealSetterFunction = std::function<void(cereal::Behavior::Builder&, double)>;

  CustomSlider(const QString &param, CerealSetterFunction cerealSetFunc, 
                           const QString &unit, const QString &title, 
                           double paramMin, double paramMax, double defaultVal, QWidget *parent = nullptr);


  QWidget *getSliderItem() {
    return sliderItem;
  }
  CerealSetterFunction cerealSetFunc;

  double paramMin;
  double paramMax;
  int sliderMin = 0;
  int sliderMax = 10000;

signals:
  void sliderReleasedWithValue(int value);

protected:
  void mouseReleaseEvent(QMouseEvent *event) override {
    QSlider::mouseReleaseEvent(event);
    emit sliderReleasedWithValue(value());
  }

private:
  void initialize();
  

  double defaultVal;
  double scaleFactor;
  QString param;
  QString title;
  QString unit;

  QWidget *sliderItem;
  QLabel *label;
  
  int sliderRange = sliderMax - sliderMin;

  QString SliderStyle = R"(
    QSlider::groove:horizontal 
      {border: none;height: 60px;background-color: #393939;border-radius: 30px;}
    QSlider::handle:horizontal 
      {background-color: #fafafa;border: none;width: 80px;height: 80px;margin-top: -10px;margin-bottom: -10px;border-radius: 40px;}
  )";
  QString lockedSliderStyle = R"(
    QSlider::groove:horizontal 
      {border: none;height: 60px;background-color: #393939;border-radius: 30px;}
    QSlider::handle:horizontal 
      {background-color: #787878;border: none;width: 80px;height: 80px;margin-top: -10px;margin-bottom: -10px;border-radius: 40px;}
  )";
  // label
  QString LabelStyle = R"(
    QLabel {
      color: #fafafa;
    }
  )";
  QString lockedLabelStyle = R"(
    QLabel {
      color: #787878;
    }
  )";
};