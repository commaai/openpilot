#pragma once

#include <QtWidgets>

#include "selfdrive/ui/ui.h"

class Sidebar : public QFrame {
  Q_OBJECT
  Q_PROPERTY(QString connectStr MEMBER connect_str NOTIFY valueChanged);
  Q_PROPERTY(QColor connectStatus MEMBER connect_status NOTIFY valueChanged);
  Q_PROPERTY(QString pandaStr MEMBER panda_str NOTIFY valueChanged);
  Q_PROPERTY(QColor pandaStatus MEMBER panda_status NOTIFY valueChanged);
  Q_PROPERTY(int tempVal MEMBER temp_val NOTIFY valueChanged);
  Q_PROPERTY(QColor tempStatus MEMBER temp_status NOTIFY valueChanged);
  Q_PROPERTY(cereal::DeviceState::NetworkType netType MEMBER net_type NOTIFY valueChanged);
  Q_PROPERTY(cereal::DeviceState::NetworkStrength netStrength MEMBER net_strength NOTIFY valueChanged);

public:
  explicit Sidebar(QWidget* parent = 0);

signals:
  void openSettings();
  void valueChanged();

public slots:
  void updateState(const UIState &s);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;

private:
  void drawMetric(QPainter &p, const QString &label, const QString &val, QColor c, int y);

  QImage home_img, settings_img;
  const QMap<cereal::DeviceState::NetworkType, QString> network_type = {
    {cereal::DeviceState::NetworkType::NONE, "--"},
    {cereal::DeviceState::NetworkType::WIFI, "WiFi"},
    {cereal::DeviceState::NetworkType::CELL2_G, "2G"},
    {cereal::DeviceState::NetworkType::CELL3_G, "3G"},
    {cereal::DeviceState::NetworkType::CELL4_G, "LTE"},
    {cereal::DeviceState::NetworkType::CELL5_G, "5G"}
  };
  const QMap<cereal::DeviceState::NetworkStrength, QImage> signal_imgs = {
    {cereal::DeviceState::NetworkStrength::UNKNOWN, QImage("../assets/images/network_0.png")},
    {cereal::DeviceState::NetworkStrength::POOR, QImage("../assets/images/network_1.png")},
    {cereal::DeviceState::NetworkStrength::MODERATE, QImage("../assets/images/network_2.png")},
    {cereal::DeviceState::NetworkStrength::GOOD, QImage("../assets/images/network_3.png")},
    {cereal::DeviceState::NetworkStrength::GREAT, QImage("../assets/images/network_4.png")},
  };

  const QRect settings_btn = QRect(50, 35, 200, 117);
  const QColor good_color = QColor(255, 255, 255);
  const QColor warning_color = QColor(218, 202, 37);
  const QColor danger_color = QColor(201, 34, 49);

  QString connect_str = "OFFLINE";
  QColor connect_status = warning_color;
  QString panda_str = "NO\nPANDA";
  QColor panda_status = warning_color;
  int temp_val = 0;
  QColor temp_status = warning_color;
  cereal::DeviceState::NetworkType net_type;
  cereal::DeviceState::NetworkStrength net_strength;
};
