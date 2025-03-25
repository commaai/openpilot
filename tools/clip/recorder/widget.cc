#include "tools/clip/recorder/widget.h"

#include "tools/clip/recorder/ffmpeg.h"

Recorder::Recorder(QObject *parent) : QObject(parent) {
  encoder = new FFmpegEncoder("/Users/trey/Desktop/out.mp4", DEVICE_SCREEN_SIZE.width(), DEVICE_SCREEN_SIZE.height(), UI_FREQ);
}

Recorder::~Recorder() {
    delete encoder;
}

void Recorder::saveFrame(const std::shared_ptr<QPixmap> &frame) {
    QMutexLocker locker(&mutex);
    frameQueue.enqueue(frame); // Add frame to queue
    if (!isProcessing) {
        isProcessing = true;
        QMetaObject::invokeMethod(this, &Recorder::processQueue, Qt::QueuedConnection);
    }
}

void Recorder::processQueue() {
    while (true) {
        std::shared_ptr<QPixmap> frame;
        {
            QMutexLocker locker(&mutex);
            if (frameQueue.isEmpty() || !keepRunning) {
                isProcessing = false;
                return;
            }
            frame = frameQueue.dequeue();
        }

        if (!encoder->writeFrame(frame->toImage().convertToFormat(QImage::Format_ARGB32_Premultiplied))) {
            fprintf(stderr, "did not write\n");
        }
    }
}

