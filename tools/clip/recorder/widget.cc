#include "tools/clip/recorder/widget.h"

#include "tools/clip/recorder/ffmpeg.h"

Recorder::Recorder(const std::string& outputFile, QObject *parent) : QObject(parent) {
  const float scale = util::getenv("SCALE", 1.0f);
  encoder = new FFmpegEncoder(outputFile, DEVICE_SCREEN_SIZE.width() * scale, DEVICE_SCREEN_SIZE.height() * scale, UI_FREQ);
}

Recorder::~Recorder() {
    keepRunning = false;  // Signal processing thread to stop
    delete encoder;
}

void Recorder::saveFrame(const std::shared_ptr<QPixmap> &frame) {
    QMutexLocker locker(&mutex);

    // Drop frame if queue is full
    if (frameQueue.size() >= MAX_QUEUE_SIZE) {
        qDebug() << "Dropping frame";
        return;
    }

    frameQueue.enqueue(frame);
    QMetaObject::invokeMethod(this, &Recorder::processQueue, Qt::QueuedConnection);
}

void Recorder::processQueue() {
    while (keepRunning) {
        std::shared_ptr<QPixmap> frame;
        {
            QMutexLocker locker(&mutex);
            if (frameQueue.isEmpty()) {
                return;
            }
            frame = frameQueue.dequeue();
        }

        if (!encoder->writeFrame(frame->toImage().convertToFormat(QImage::Format_ARGB32))) {
            fprintf(stderr, "did not write\n");
        }
    }
}

