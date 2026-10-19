# Safety

openpilot is an Adaptive Cruise Control (ACC) and Automated Lane Centering (ALC) system.
It is a SAE Level 2 driver assistance system. openpilot is not a self-driving system and does
not make the vehicle autonomous. It does not replace the driver's attention, judgment, and
need to control the vehicle.

The driver is responsible for operating the vehicle safely at all times. The driver must
pay attention to the road and surrounding traffic and be ready to take immediate control
of the vehicle at all times.

To assist the driver in maintaining alertness, openpilot includes a driver monitoring feature
that alerts when it detects driver distraction. See [LIMITATIONS.md](LIMITATIONS.md) before use.

Driver alertness is necessary, but not sufficient, for openpilot to be used safely.
openpilot is provided with no warranty of fitness for any purpose.

openpilot is developed in good faith to be compliant with FMVSS requirements and to follow
industry standards of safety for Level 2 Driver Assistance Systems. In particular, we observe
ISO26262 guidelines, including those from [pertinent documents](https://www.nhtsa.gov/sites/nhtsa.dot.gov/files/documents/13498a_812_573_alcsystemreport.pdf)
released by NHTSA. In addition, we impose strict coding guidelines (like [MISRA C : 2012](https://www.misra.org.uk/what-is-misra/))
on parts of openpilot that are safety relevant. We also perform software-in-the-loop,
hardware-in-the-loop, and in-vehicle tests before each software release.

Following Hazard and Risk Analysis and FMEA, at a very high level, we have designed openpilot
ensuring two main safety requirements.

1. The driver must always be able to immediately retake manual control of the vehicle,
   by stepping on the brake pedal or by pressing the cancel button.
2. The vehicle must not alter its trajectory too quickly for the driver to safely
   react. This means that while the system is engaged, the actuators are constrained
   to operate within reasonable limits[^1].

For additional safety implementation details, refer to [panda safety model](https://github.com/commaai/panda#safety-model). For vehicle specific implementation of the safety concept, refer to [opendbc/safety/safety](https://github.com/commaai/opendbc/tree/master/opendbc/safety/safety).

[^1]: For these actuator limits we observe ISO11270 and ISO15622. Lateral limits described there translate to 0.9 seconds of maximum actuation to achieve a 1m lateral deviation.

---

### Forks of openpilot

* Do not disable or nerf [driver monitoring](https://github.com/commaai/openpilot/tree/master/openpilot/selfdrive/monitoring)
* Do not disable or nerf [excessive actuation checks](https://github.com/commaai/openpilot/tree/master/openpilot/selfdrive/selfdrived/helpers.py)
* If your fork modifies any of the code in `opendbc/safety/`:
   * your fork cannot use the openpilot trademark
   * your fork must preserve the full [safety test suite](https://github.com/commaai/opendbc/tree/master/opendbc/safety/tests) and all tests must pass, including any new coverage required by the fork's changes

Failure to comply with these standards will get you and your users banned from comma.ai servers.

**comma.ai strongly discourages the use of openpilot forks with safety code either missing or not fully meeting the above requirements.**
