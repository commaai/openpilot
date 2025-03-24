#pragma once
#include <selfdrive/ui/qt/onroad/onroad_home.h>

#include "recorder/widget.h"


class Application {
public:
    Application(int argc, char* argv[]);
    ~Application();
    int exec() const;
    void close() const;
private:
    QApplication *app;
    QThread *recorderThread = nullptr;
    Recorder *recorder = nullptr;
    OnroadWindow *window;
};
