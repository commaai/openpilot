#pragma once
#include <selfdrive/ui/qt/onroad/onroad_home.h>
#include <tools/replay/replay.h>
#include <QThread>

#include "recorder/widget.h"


class Application {
public:
    Application(int argc, char* argv[]);
    ~Application();
    int exec() const;
    void close() const;

private:
    void initReplay();
    void startReplay();

    QApplication *app;
    QThread *recorderThread = nullptr;
    Recorder *recorder = nullptr;
    OnroadWindow *window;

    // Replay related members
    std::unique_ptr<Replay> replay;
    QThread *replayThread = nullptr;
    bool replayRunning = false;
};
