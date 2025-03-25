#pragma once

#include <QImage>
#include <QObject>
#include <QQueue>
#include <selfdrive/ui/ui.h>
#include <selfdrive/ui/qt/qt_window.h>

#include "ffmpeg.h"

class Recorder : public QObject {
    Q_OBJECT

public:
    Recorder(QObject *parent = nullptr);
    ~Recorder() override;

public slots:
    void saveFrame(const std::shared_ptr<QPixmap> &frame);

private:
    FFmpegEncoder *encoder;
    QQueue<std::shared_ptr<QPixmap>> frameQueue;
    QMutex mutex;
    bool isProcessing = false;
    bool keepRunning = true;
    void processQueue();
};
