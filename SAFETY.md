openpilot Safety
======

openpilot is an Adaptive Cruise Control (ACC) and Automated Lane Centering (ALC) system. 
Like other ACC and ALC systems, openpilot is a failsafe passive system and it requires the
driver to be alert and to pay attention at all times.

In order to enforce driver alertness, openpilot includes a driver monitoring feature
that alerts the driver when distracted.

However, even with an attentive driver, we must make further efforts for the system to be
safe. We repeat, **driver alertness is necessary, but not sufficient, for openpilot to be
used safely** and openpilot is provided with no warranty of fitness for any purpose.

openpilot is developed in good faith to be compliant with FMVSS requirements and to follow
industry standards of safety for Level 2 Driver Assistance Systems. In particular, we observe
ISO26262 guidelines, including those from [pertinent documents](https://www.nhtsa.gov/sites/nhtsa.dot.gov/files/documents/13498a_812_573_alcsystemreport.pdf)
released by NHTSA. In addition, we impose strict coding guidelines (like [MISRA C : 2012](https://www.misra.org.uk/MISRAHome/MISRAC2012/tabid/196/Default.aspx))
on parts of openpilot that are safety relevant. We also perform software-in-the-loop,
hardware-in-the-loop and in-vehicle tests before each software release.

Following Hazard and Risk Analysis and FMEA, at a very high level, we have designed openpilot
ensuring two main safety requirements.

1. The driver must always be capable to immediately retake manual control of the vehicle, 
   by stepping on either pedal or by pressing the cancel button.
2. The vehicle must not alter its trajectory too quickly for the driver to safely
   react. This means that while the system is engaged, the actuators are constrained
   to operate within reasonable limits.

For vehicle specific implementation of the safety concept, refer to `panda/board/safety/`.

**Extra note**: comma.ai strongly discourages the use of openpilot forks with safety code either missing or
  not fully meeting the above requirements.
