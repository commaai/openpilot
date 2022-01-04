#pragma once

class Wakeable {

  protected:
    bool awake = false;
    int interactive_timeout = 0;
    
    virtual void setAwake(bool on) = 0;
};
