#pragma once

#include <optional>
#include <vector>

#include <QDateTime>
#include <QFileSystemWatcher>
#include <QPainter>
#include <QPixmap>
#include <QSurfaceFormat>
#include <QWidget>

#include "cereal/gen/cpp/car.capnp.h"
#include "common/params.h"

QString getVersion();
QString getBrand();
QString getUserAgent();
std::optional<QString> getDongleId();
QMap<QString, QString> getSupportedLanguages();
void setQtSurfaceFormat();
void sigTermHandler(int s);
QString timeAgo(const QDateTime &date);
void swagLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
void initApp(int argc, char *argv[], bool disable_hidpi = true);
QWidget* topWidget(QWidget* widget);
QPixmap loadPixmap(const QString &fileName, const QSize &size = {}, Qt::AspectRatioMode aspectRatioMode = Qt::KeepAspectRatio);
QPixmap bootstrapPixmap(const QString &id);

void drawRoundedRect(QPainter &painter, const QRectF &rect, qreal xRadiusTop, qreal yRadiusTop, qreal xRadiusBottom, qreal yRadiusBottom);
QColor interpColor(float xv, std::vector<float> xp, std::vector<QColor> fp);
bool hasLongitudinalControl(const cereal::CarParams::Reader &car_params);

struct InterFont : public QFont {
  InterFont(int pixel_size, QFont::Weight weight = QFont::Normal) : QFont("Inter") {
    setPixelSize(pixel_size);
    setWeight(weight);
  }
};

class ParamWatcher : public QObject {
  Q_OBJECT

public:
  ParamWatcher(QObject *parent);
  void addParam(const QString &param_name);

signals:
  void paramChanged(const QString &param_name, const QString &param_value);

private:
  void fileChanged(const QString &path);

  QFileSystemWatcher *watcher;
  QHash<QString, QString> params_hash;
  Params params;
};
