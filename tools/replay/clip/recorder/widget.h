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
    Recorder(QObject *parent = nullptr) : QObject(parent) {}
    ~Recorder() override;

public slots:
    void saveFrame(QImage *frame);
    void stop();

private:
    FFmpegEncoder encoder = FFmpegEncoder("/Users/trey/Desktop/out.mp4", DEVICE_SCREEN_SIZE.width(), DEVICE_SCREEN_SIZE.height(), UI_FREQ);
    QQueue<QImage *> frameQueue;
    QMutex mutex;
    bool isProcessing = false;
    bool keepRunning = true;
    void processQueue();
};
