#pragma once

#include <QObject>

// Forward declaration
class UIState;

class Wakeable {

  public:
    virtual void resetInteractiveTimout() = 0; 

  protected:
    bool awake = false;
    int interactive_timeout = 0;
    
    virtual void setAwake(bool on);
    virtual void updateWakefulness(const UIState &s) = 0;
    virtual void emitDisplayPowerChanged(bool on) = 0;
};
