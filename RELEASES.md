Version 0.3.2  (2017-05-22)
===========================
  * Minor stability bugfixes
  * Added metrics and rear view mirror disable to settings
  * Update model with more crowdsourced data

Version 0.3.1  (2017-05-17)
===========================
  * visiond stability bugfix
  * Add logging for angle and flashing

Version 0.3.0  (2017-05-12)
===========================
  * Add CarParams struct to improve the abstraction layer
  * Refactor visiond IPC to support multiple clients
  * Add raw GPS and beginning support for navigation
  * Improve model in visiond using crowdsourced data
  * Add improved system logging to diagnose instability
  * Rewrite baseui in React Native
  * Moved calibration to the cloud

Version 0.2.9  (2017-03-01)
===========================
  * Retain compatibility with NEOS v1

Version 0.2.8  (2017-02-27)
===========================
  * Fix bug where frames were being dropped in minute 71

Version 0.2.7  (2017-02-08)
===========================
  * Better performance and pictures at night
  * Fix ptr alignment issue in boardd
  * Fix brake error light, fix crash if too cold

Version 0.2.6  (2017-01-31)
===========================
  * Fix bug in visiond model execution

Version 0.2.5  (2017-01-30)
===========================
  * Fix race condition in manager

Version 0.2.4  (2017-01-27)
===========================
  * OnePlus 3T support
  * Enable installation as NEOS app
  * Various minor bugfixes

Version 0.2.3  (2017-01-11)
===========================
  * Reduce space usage by 80%
  * Add better logging
  * Add Travis CI

Version 0.2.2  (2017-01-10)
===========================
  * Board triggers started signal on CAN messages
  * Improved autoexposure
  * Handle out of space, improve upload status

Version 0.2.1  (2016-12-14)
===========================
  * Performance improvements, removal of more numpy
  * Fix boardd process priority
  * Make counter timer reset on use of steering wheel

Version 0.2  (2016-12-12)
=========================
  * Car/Radar abstraction layers have shipped, see cereal/car.capnp
  * controlsd has been refactored
  * Shipped plant model and testing maneuvers
  * visiond exits more gracefully now
  * Hardware encoder in visiond should always init
  * ui now turns off the screen after 30 seconds
  * Switch to openpilot release branch for future releases
  * Added preliminary Docker container to run tests on PC

Version 0.1  (2016-11-29)
=========================
  * Initial release of openpilot
  * Adaptive cruise control is working
  * Lane keep assist is working
  * Support for Acura ILX 2016 with AcuraWatch Plus
  * Support for Honda Civic 2016 Touring Edition

