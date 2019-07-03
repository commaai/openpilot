Welcome to chffrplus
======

[chffrplus](https://github.com/commaai/chffrplus) is an open source dashcam.

This is the shipping reference software for the comma EON Dashcam DevKit. It keeps many of the niceities of [openpilot](https://github.com/commaai/openpilot), like high quality sensors, great camera, and good autostart and stop. Though unlike openpilot, it cannot control your car. chffrplus can interface with your car through a [panda](https://shop.comma.ai/products/panda-obd-ii-dongle), but just like our dashcam app [chffr](https://getchffr.com/), it is read only.

It integrates with the rest of the comma ecosystem, so you can view your drives on the [chffr](https://getchffr.com/) app for Android or iOS, and reverse engineer your car with [cabana](https://community.comma.ai/cabana/?demo=1).


Hardware
------

Right now chffrplus supports the [EON Dashcam DevKit](https://shop.comma.ai/products/eon-dashcam-devkit) for hardware to run on.

Install chffrplus on a EON device by entering ``https://chffrplus.comma.ai`` during NEOS setup.


User Data / chffr Account / Crash Reporting
------

By default chffrplus creates an account and includes a client for chffr, our dashcam app.

It's open source software, so you are free to disable it if you wish. 

It logs the road facing camera, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
It does not log the user facing camera or the microphone.

By using it, you agree to [our privacy policy](https://beta.comma.ai/privacy.html). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma.ai. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma.ai for the use of this data.


Licensing
------

chffrplus is released under the MIT license.

