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
    Recorder(const std::string& outputFile, QObject *parent = nullptr);
    ~Recorder() override;

public slots:
    void saveFrame(const std::shared_ptr<QPixmap> &frame);

private:
    static constexpr int MAX_QUEUE_SIZE = 30;  // Limit queue size to prevent memory growth
    FFmpegEncoder *encoder;
    QQueue<std::shared_ptr<QPixmap>> frameQueue;
    QMutex mutex;
    QAtomicInt isProcessing{0};  // Use atomic for thread safety
    bool keepRunning = true;
    void processQueue();
};
