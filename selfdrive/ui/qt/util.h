#pragma once

#include <optional>

#include <QDateTime>
#include <QLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QSurfaceFormat>
#include <QWidget>

QString getBrand();
QString getBrandVersion();
QString getUserAgent();
std::optional<QString> getDongleId();
void configFont(QPainter &p, const QString &family, int size, const QString &style);
void clearLayout(QLayout* layout);
void setQtSurfaceFormat();
QString timeAgo(const QDateTime &date);
void swagLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
void initApp();
QWidget* topWidget (QWidget* widget);


// convenience class for wrapping layouts
class LayoutWidget : public QWidget {
  Q_OBJECT

public:
  LayoutWidget(QLayout *l, QWidget *parent = nullptr) : QWidget(parent) {
    setLayout(l);
  };
};

class ClickableWidget : public QWidget {
  Q_OBJECT

public:
  ClickableWidget(QWidget *parent = nullptr);

protected:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void paintEvent(QPaintEvent *) override;

signals:
  void clicked();
};
