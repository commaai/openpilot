#pragma once

#include <optional>

#include <QDateTime>
#include <QLayout>
#include <QPainter>
#include <QPixmap>
#include <QSurfaceFormat>
#include <QWidget>

QString getVersion();
QString getBrand();
QString getBrandVersion();
QString getUserAgent();
std::optional<QString> getDongleId();
QMap<QString, QString> getSupportedLanguages();
void configFont(QPainter &p, const QString &family, int size, const QString &style);
void clearLayout(QLayout* layout);
void setQtSurfaceFormat();
QString timeAgo(const QDateTime &date);
void swagLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
void initApp(int argc, char *argv[]);
QWidget* topWidget (QWidget* widget);
QPixmap loadPixmap(const QString &fileName, const QSize &size = {}, Qt::AspectRatioMode aspectRatioMode = Qt::KeepAspectRatio);

QRect getTextRect(QPainter &p, int flags, const QString &text);
void drawRoundedRect(QPainter &painter, const QRectF &rect, qreal xRadiusTop, qreal yRadiusTop, qreal xRadiusBottom, qreal yRadiusBottom);
QColor interpColor(float xv, std::vector<float> xp, std::vector<QColor> fp);
