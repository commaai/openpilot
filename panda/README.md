Welcome to panda
======

[panda](http://github.com/commaai/panda) is the nicest universal car interface ever.

<a href="https://comma.ai/shop/products/panda-obd-ii-dongle"><img src="https://github.com/commaai/panda/blob/master/panda.png">

<img src="https://github.com/commaai/panda/blob/master/buy.png"></a>

It supports 3x CAN, 2x LIN, and 1x GMLAN. It also charges a phone. On the computer side, it has both USB and Wi-Fi.

It uses an [STM32F413](http://www.st.com/en/microcontrollers/stm32f413-423.html?querycriteria=productId=LN2004) for low level stuff and an [ESP8266](https://en.wikipedia.org/wiki/ESP8266) for Wi-Fi. They are connected over high speed SPI, so the panda is actually capable of dumping the full contents of the busses over Wi-Fi, unlike every other dongle on amazon. ELM327 is weak, panda is strong.

It is 2nd gen hardware, reusing code and parts from the [NEO](https://github.com/commaai/neo) interface board.

[![CircleCI](https://circleci.com/gh/commaai/panda.svg?style=svg)](https://circleci.com/gh/commaai/panda)

Usage (Python)
------

To install the library:
```
# pip install pandacan
```

See [this class](https://github.com/commaai/panda/blob/master/python/__init__.py#L80) for how to interact with the panda.

For example, to receive CAN messages:
```
>>> from panda import Panda
>>> panda = Panda()
>>> panda.can_recv()
```
And to send one on bus 0:
```
>>> panda.can_send(0x1aa, "message", 0)
```
Find user made scripts on the [wiki](https://community.comma.ai/wiki/index.php/Panda_scripts)

Note that you may have to setup [udev rules](https://community.comma.ai/wiki/index.php/Panda#Linux_udev_rules) for Linux, such as
```
sudo -i
echo 'SUBSYSTEMS=="usb", ATTR{idVendor}=="bbaa", ATTR{idProduct}=="ddcc", MODE:="0666"' > /etc/udev/rules.d/11-panda.rules
exit
```

Usage (JavaScript)
-------

See [PandaJS](https://github.com/commaai/pandajs)


Software interface support
------

As a universal car interface, it should support every reasonable software interface.

- [User space](https://github.com/commaai/panda/tree/master/python)
- [socketcan in kernel](https://github.com/commaai/panda/tree/master/drivers/linux) (alpha)
- [ELM327](https://github.com/commaai/panda/blob/master/boardesp/elm327.c)
- [Windows J2534](https://github.com/commaai/panda/tree/master/drivers/windows)

Directory structure
------

- board      -- Code that runs on the STM32
- boardesp   -- Code that runs on the ESP8266
- drivers    -- Drivers (not needed for use with python)
- python     -- Python userspace library for interfacing with the panda
- tests      -- Tests and helper programs for panda

Programming (over USB)
------

[Programming the Board (STM32)](board/README.md)

[Programming the ESP](boardesp/README.md)


Debugging
------

To print out the serial console from the STM32, run tests/debug_console.py

To print out the serial console from the ESP8266, run PORT=1 tests/debug_console.py

Safety Model
------

When a panda powers up, by default it's in `SAFETY_NOOUTPUT` mode. While in no output mode, the buses are also forced to be silent. In order to send messages, you have to select a safety mode. Currently, setting safety modes is only supported over USB.

Safety modes optionally supports `controls_allowed`, which allows or blocks a subset of messages based on a customizable state in the board.

Code Rigor
------
When compiled from an [EON Dev Kit](https://comma.ai/shop/products/eon-gold-dashcam-devkit), the panda FW is configured and optimized (at compile time) for its use in
conjuction with [openpilot](https://github.com/commaai/openpilot). The panda FW, through its safety model, provides and enforces the
[openpilot Safety](https://github.com/commaai/openpilot/blob/devel/SAFETY.md). Due to its critical function, it's important that the application code rigor within the `board` folder is held to high standards.

These are the [CI regression tests](https://circleci.com/gh/commaai/panda) we have in place:
* A generic static code analysis is performed by [Cppcheck](https://github.com/danmar/cppcheck/).
* In addition, [Cppcheck](https://github.com/danmar/cppcheck/) has a specific addon to check for [MISRA C:2012](https://www.misra.org.uk/MISRAHome/MISRAC2012/tabid/196/Default.aspx) violations. See [current coverage](https://github.com/commaai/panda/blob/master/tests/misra/coverage_table).
* Compiler options are relatively strict: the flags `-Wall -Wextra -Wstrict-prototypes -Werror` are enforced on board and pedal makefiles.
* The [safety logic](https://github.com/commaai/panda/tree/master/board/safety) is tested and verified by [unit tests](https://github.com/commaai/panda/tree/master/tests/safety) for each supported car variant.
* A recorded drive for each supported car variant is [replayed through the safety logic](https://github.com/commaai/panda/tree/master/tests/safety_replay)
to ensure that the behavior remains unchanged.
* An internal Hardware-in-the-loop test, which currently only runs on pull requests opened by comma.ai's organization members, verifies the following functionalities:
    * compiling the code in various configuration and flashing it both through USB and WiFi.
    * Receiving, sending and forwarding CAN messages on all buses, over USB and WiFi.

In addition, we run [Pylint](https://www.pylint.org/) and [Flake8](https://github.com/PyCQA/flake8) linters on all python files within the panda repo.

Hardware
------

Check out the hardware [guide](https://github.com/commaai/panda/blob/master/docs/guide.pdf)

Licensing
------

panda software is released under the MIT license unless otherwise specified.
