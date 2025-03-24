#include "tools/replay/clip/recorder/widget.h"

#include "tools/replay/clip/recorder/ffmpeg.h"

Recorder::Recorder(QObject *parent) : QObject(parent) {
  encoder = new FFmpegEncoder("/Users/trey/Desktop/out.mp4", DEVICE_SCREEN_SIZE.width(), DEVICE_SCREEN_SIZE.height(), UI_FREQ);
}

Recorder::~Recorder() {
    fprintf(stderr, "closing\n");
    delete encoder;
    QObject::~QObject();
}

void Recorder::saveFrame(QImage *frame) {
    QMutexLocker locker(&mutex);
    frameQueue.enqueue(frame); // Add frame to queue
    if (!isProcessing) {
        isProcessing = true;
        QMetaObject::invokeMethod(this, &Recorder::processQueue, Qt::QueuedConnection);
    }
}

void Recorder::processQueue() {
    while (true) {
        QImage *frame;
        {
            QMutexLocker locker(&mutex);
            if (frameQueue.isEmpty() || !keepRunning) {
                isProcessing = false;
                return;
            }
            frame = frameQueue.dequeue();
        }

        if (!encoder->writeFrame(frame->convertToFormat(QImage::Format_ARGB32))) {
            fprintf(stderr, "did not write\n");
        }

        delete frame;
    }
}

void Recorder::stop() {
    QMutexLocker locker(&mutex);
    keepRunning = false;
}

