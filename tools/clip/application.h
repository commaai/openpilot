#pragma once
#include <selfdrive/ui/qt/onroad/onroad_home.h>
#include <tools/replay/replay.h>
#include <QThread>

#include "recorder/widget.h"


class Application : public QObject {
    Q_OBJECT
public:
    Application(int argc, char* argv[], QObject *parent = nullptr);
    ~Application();
    int exec() const;
    void close() const;

private:
    void initReplay(const std::string& route);
    void startReplay(int start = 0);

    QApplication *app;
    QThread *recorderThread = nullptr;
    Recorder *recorder = nullptr;
    OnroadWindow *window;

    // Replay related members
    std::unique_ptr<Replay> replay;
    QThread *replayThread = nullptr;
    bool replayRunning = false;
};
