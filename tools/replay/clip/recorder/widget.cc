#include "tools/replay/clip/recorder/widget.h"

#include "tools/replay/clip/recorder/ffmpeg.h"

Recorder::~Recorder() {
    fprintf(stderr, "closing\n");
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
        // Save the frame (this runs in the worker thread)
        static int frameCount = 0;
        if (frameCount == 0 && !encoder.startRecording()) {
            fprintf(stderr, "failed to start record\n");
        }
        fprintf(stderr, "processing frame %d: %p\n", frameCount++, &frame);
        if (!encoder.writeFrame(frame->convertToFormat(QImage::Format_ARGB32_Premultiplied))) {
            fprintf(stderr, "did not write\n");
        }
    }
}

void Recorder::stop() {
    QMutexLocker locker(&mutex);
    keepRunning = false;
}

