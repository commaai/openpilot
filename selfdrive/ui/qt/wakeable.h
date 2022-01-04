#pragma once

#include <QObject>

class Wakeable {

  public:
    virtual void resetInteractiveTimout(); 

  protected:
    bool awake = false;
    int interactive_timeout = 0;
    
    virtual void setAwake(bool on) = 0;
    virtual void updateWakefulness(const QObject &o) = 0;

};
