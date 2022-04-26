Version 0.8.14 (2022-0X-XX)
========================
 * New driving model
   * bigmodel
 * New driver monitoring model
 * New lateral controller
   * Smoother control
   * Simplified tuning
   * Initially used on TSS2 Corolla and TSS-P Rav4
 * comma body support
 * Audi RS3 support thanks to jyoung8607!
 * Hyundai Ioniq Plug-in Hybrid 2019 support thanks to sunnyhaibin!
 * Hyundai Tucson Diesel 2019 support thanks to sunnyhaibin!
 * Toyota Alphard Hybrid 2021 support
 * Toyota Avalon Hybrid 2022 support

Version 0.8.13 (2022-02-18)
========================
 * Improved driver monitoring
   * Retuned driver pose learner for relaxed driving positions
   * Added reliance on driving model to be more scene adaptive
   * Matched strictness between comma two and comma three
 * Improved performance in turns by compensating for the road bank angle
 * Improved camera focus on the comma two
 * AGNOS 4
   * ADB support
   * improved cell auto configuration
 * NEOS 19
   * package updates
   * stability improvements
 * Subaru ECU firmware fingerprinting thanks to martinl!
 * Hyundai Santa Fe Plug-in Hybrid 2022 support thanks to sunnyhaibin!
 * Mazda CX-5 2022 support thanks to Jafaral!
 * Subaru Impreza 2020 support thanks to martinl!
 * Toyota Avalon 2022 support thanks to sshane!
 * Toyota Prius v 2017 support thanks to CT921!
 * Volkswagen Caravelle 2020 support thanks to jyoung8607!

Version 0.8.12 (2021-12-15)
========================
 * New driving model
   * Improved behavior around exits
   * Better pose accuracy at high speeds, allowing max speed of 90mph
   * Fully incorporated comma three data into all parts of training stack
 * Improved follow distance
 * Better longitudinal policy, especially in low speed traffic
 * New alert sounds
 * AGNOS 3
   * Display burn in mitigation
   * Improved audio amplifier configuration
   * System reliability improvements
   * Update Python to 3.8.10
 * Raw logs upload moved to connect.comma.ai
 * Fixed HUD alerts on newer Honda Bosch thanks to csouers!
 * Audi Q3 2020-21 support thanks to jyoung8607!
 * Lexus RC 2020 support thanks to ErichMoraga!

Version 0.8.11 (2021-11-29)
========================
 * Support for CAN FD on the red panda
 * Support for an external panda on the comma three
 * Navigation: Show more detailed instructions when approaching maneuver
 * Fixed occasional steering faults on GM cars thanks to jyoung8607!
 * Nissan ECU firmware fingerprinting thanks to robin-reckmann, martinl, and razem-io!
 * Cadillac Escalade ESV 2016 support thanks to Gibby!
 * Genesis G70 2020 support thanks to tecandrew!
 * Hyundai Santa Fe Hybrid 2022 support thanks to sunnyhaibin!
 * Mazda CX-9 2021 support thanks to Jacar!
 * Volkswagen Polo 2020 support thanks to jyoung8607!
 * Volkswagen T-Roc 2021 support thanks to jyoung8607!

Version 0.8.10 (2021-11-01)
========================
 * New driving model
   * Trained on one million minutes!!!
   * Fixed lead training making lead predictions significantly more accurate
   * Fixed several localizer dataset bugs and loss function bugs, overall improved accuracy
 * New driver monitoring model
   * Trained on latest data from both comma two and comma three
   * Increased model field of view by 40% on comma three
   * Improved model stability on masked users
   * Improved pose prediction with reworked ground-truth stack
 * Lateral and longitudinal planning MPCs now in ACADOS
 * Combined longitudinal MPCs
   * All longitudinal planning now happens in a single MPC system
   * Fixed instability in MPC problem to prevent sporadic CPU usage
 * AGNOS 2: minor stability improvements and builder repo open sourced
 * tools: new and improved replay thanks to deanlee!
 * Moved community-supported cars outside of the Community Features toggle
 * Improved FW fingerprinting reliability for Hyundai/Kia/Genesis
 * Added prerequisites for longitudinal control on Hyundai/Kia/Genesis and Honda Bosch
 * Audi S3 2015 support thanks to jyoung8607!
 * Honda Freed 2020 support thanks to belm0!
 * Hyundai Ioniq Hybrid 2020-2022 support thanks to sunnyhaibin!
 * Hyundai Santa Fe 2022 support thanks to sunnyhaibin!
 * Kia K5 2021 support thanks to sunnyhaibin!
 * Škoda Kamiq 2021 support thanks to jyoung8607!
 * Škoda Karoq 2019 support thanks to jyoung8607!
 * Volkswagen Arteon 2021 support thanks to jyoung8607!
 * Volkswagen California 2021 support thanks to jyoung8607!
 * Volkswagen Taos 2022 support thanks to jyoung8607!

Version 0.8.9 (2021-09-14)
========================
 * Improved fan control on comma three
 * AGNOS 1.5: improved stability
 * Honda e 2020 support

Version 0.8.8 (2021-08-27)
========================
 * New driving model with improved laneless performance
   * Trained on 5000+ hours of diverse driving data from 3000+ users in 40+ countries
   * Better anti-cheating methods during simulator training ensure the model hugs less when in laneless mode
   * All new desire ground-truthing stack makes the model better at lane changes
 * New driver monitoring model: improved performance on comma three
 * NEOS 18 for comma two: update packages
 * AGNOS 1.3 for comma three: fix display init at high temperatures
 * Improved auto-exposure on comma three
 * Improved longitudinal control on Honda Nidec cars
 * Hyundai Kona Hybrid 2020 support thanks to haram-KONA!
 * Hyundai Sonata Hybrid 2021 support thanks to Matt-Wash-Burn!
 * Kia Niro Hybrid 2021 support thanks to tetious!

Version 0.8.7 (2021-07-31)
========================
 * comma three support!
 * Navigation alpha for the comma three!
 * Volkswagen T-Cross 2021 support thanks to jyoung8607!

Version 0.8.6 (2021-07-21)
========================
 * Revamp lateral and longitudinal planners
   * Refactor planner output API to be more readable and verbose
   * Planners now output desired trajectories for speed, acceleration, curvature, and curvature rate
   * Use MPC for longitudinal planning when no lead car is present, makes accel and decel smoother
 * Remove "CHECK DRIVER FACE VISIBILITY" warning
 * Fixed cruise fault on some TSS2.5 Camrys and international Toyotas
 * Hyundai Elantra Hybrid 2021 support thanks to tecandrew!
 * Hyundai Ioniq PHEV 2020 support thanks to YawWashout!
 * Kia Niro Hybrid 2019 support thanks to jyoung8607!
 * Škoda Octavia RS 2016 support thanks to jyoung8607!
 * Toyota Alphard 2020 support thanks to belm0!
 * Volkswagen Golf SportWagen 2015 support thanks to jona96!
 * Volkswagen Touran 2017 support thanks to jyoung8607!

Version 0.8.5 (2021-06-11)
========================
 * NEOS update: improved reliability and stability with better voltage regulator configuration
 * Smart model-based Forward Collision Warning
 * CAN-based fingerprinting moved behind community features toggle
 * Improved longitudinal control on Toyotas with a comma pedal
 * Improved auto-brightness using road-facing camera
 * Added "Software" settings page with updater controls
 * Audi Q2 2018 support thanks to jyoung8607!
 * Hyundai Elantra 2021 support thanks to CruiseBrantley!
 * Lexus UX Hybrid 2019-2020 support thanks to brianhaugen2!
 * Toyota Avalon Hybrid 2019 support thanks to jbates9011!
 * SEAT Leon 2017 & 2020 support thanks to jyoung8607!
 * Škoda Octavia 2015 & 2019 support thanks to jyoung8607!

Version 0.8.4 (2021-05-17)
========================
 * Delay controls start until system is ready
 * Fuzzy car identification, enabled with Community Features toggle
 * Localizer optimized for increased precision and less CPU usage
 * Retuned lateral control to be more aggressive when model is confident
 * Toyota Mirai 2021 support
 * Lexus NX 300 2020 support thanks to goesreallyfast!
 * Volkswagen Atlas 2018-19 support thanks to jyoung8607!

Version 0.8.3 (2021-04-01)
========================
 * New model
   * Trained on new diverse dataset from 2000+ users from 30+ countries
   * Trained with improved segnet from the comma-pencil community project
   * 🥬 Dramatically improved end-to-end lateral performance 🥬
 * Toggle added to disable the use of lanelines
 * NEOS update: update packages and support for new UI
 * New offroad UI based on Qt
 * Default SSH key only used for setup
 * Kia Ceed 2019 support thanks to ZanZaD13!
 * Kia Seltos 2021 support thanks to speedking456!
 * Added support for many Volkswagen and Škoda models thanks to jyoung8607!

Version 0.8.2 (2021-02-26)
========================
 * Use model points directly in MPC (no more polyfits), making lateral planning more accurate
 * Use model heading prediction for smoother lateral control
 * Smarter actuator delay compensation
 * Improve qcamera resolution for improved video in explorer and connect
 * Adjust maximum engagement speed to better fit the model's training distribution
 * New driver monitoring model trained with 3x more diverse data
 * Improved face detection with masks
 * More predictable DM alerts when visibility is bad
 * Rewritten video streaming between openpilot processes
 * Improved longitudinal tuning on TSS2 Corolla and Rav4 thanks to briskspirit!
 * Audi A3 2015 and 2017 support thanks to keeleysam!
 * Nissan Altima 2020 support thanks to avolmensky!
 * Lexus ES Hybrid 2018 support thanks to TheInventorMan!
 * Toyota Camry Hybrid 2021 support thanks to alancyau!

Version 0.8.1 (2020-12-21)
========================
 * Original EON is deprecated, upgrade to comma two
 * Better model performance in heavy rain
 * Better lane positioning in turns
 * Fixed bug where model would cut turns on empty roads at night
 * Fixed issue where some Toyotas would not completely stop thanks to briskspirit!
 * Toyota Camry 2021 with TSS2.5 support
 * Hyundai Ioniq Electric 2020 support thanks to baldwalker!

Version 0.8.0 (2020-11-30)
========================
 * New driving model: fully 3D and improved cut-in detection
 * UI draws 2 road edges, 4 lanelines and paths in 3D
 * Major fixes to cut-in detection for openpilot longitudinal
 * Grey panda is no longer supported, upgrade to comma two or black panda
 * Lexus NX 2018 support thanks to matt12eagles!
 * Kia Niro EV 2020 support thanks to nickn17!
 * Toyota Prius 2021 support thanks to rav4kumar!
 * Improved lane positioning with uncertain lanelines, wide lanes and exits
 * Improved lateral control for Prius and Subaru

Version 0.7.10 (2020-10-29)
========================
 * Grey panda is deprecated, upgrade to comma two or black panda
 * NEOS update: update to Python 3.8.2 and lower CPU frequency
 * Improved thermals due to reduced CPU frequency
 * Update SNPE to 1.41.0
 * Reduced offroad power consumption
 * Various system stability improvements
 * Acura RDX 2020 support thanks to csouers!

Version 0.7.9 (2020-10-09)
========================
 * Improved car battery power management
 * Improved updater robustness
 * Improved realtime performance
 * Reduced UI and modeld lags
 * Increased torque on 2020 Hyundai Sonata and Palisade

Version 0.7.8 (2020-08-19)
========================
 * New driver monitoring model: improved face detection and better compatibility with sunglasses
 * Download NEOS operating system updates in the background
 * Improved updater reliability and responsiveness
 * Hyundai Kona 2020, Veloster 2019, and Genesis G70 2018 support thanks to xps-genesis!

Version 0.7.7 (2020-07-20)
========================
 * White panda is no longer supported, upgrade to comma two or black panda
 * Improved vehicle model estimation using high precision localizer
 * Improved thermal management on comma two
 * Improved autofocus for road-facing camera
 * Improved noise performance for driver-facing camera
 * Block lane change start using blindspot monitor on select Toyota, Hyundai, and Subaru
 * Fix GM ignition detection
 * Code cleanup and smaller release sizes
 * Hyundai Sonata 2020 promoted to officially supported car
 * Hyundai Ioniq Electric Limited 2019 and Ioniq SE 2020 support thanks to baldwalker!
 * Subaru Forester 2019 and Ascent 2019 support thanks to martinl!

Version 0.7.6.1 (2020-06-16)
========================
 * Hotfix: update kernel on some comma twos (orders #8570-#8680)

Version 0.7.6 (2020-06-05)
========================
 * White panda is deprecated, upgrade to comma two or black panda
 * 2017 Nissan X-Trail, 2018-19 Leaf and 2019 Rogue support thanks to avolmensky!
 * 2017 Mazda CX-5 support in dashcam mode thanks to Jafaral!
 * Huge CPU savings in modeld by using thneed!
 * Lots of code cleanup and refactors

Version 0.7.5 (2020-05-13)
========================
 * Right-Hand Drive support for both driving and driver monitoring!
 * New driving model: improved at sharp turns and lead speed estimation
 * New driver monitoring model: overall improvement on comma two
 * Driver camera preview in settings to improve mounting position
 * Added support for many Hyundai, Kia, Genesis models thanks to xx979xx!
 * Improved lateral tuning for 2020 Toyota Rav 4 (hybrid)

Version 0.7.4 (2020-03-20)
========================
 * New driving model: improved lane changes and lead car detection
 * Improved driver monitoring model: improve eye detection
 * Improved calibration stability
 * Improved lateral control on some 2019 and 2020 Toyota Prius
 * Improved lateral control on VW Golf: 20% more steering torque
 * Fixed bug where some 2017 and 2018 Toyota C-HR would use the wrong steering angle sensor
 * Support for Honda Insight thanks to theantihero!
 * Code cleanup in car abstraction layers and ui

Version 0.7.3 (2020-02-21)
========================
 * Support for 2020 Highlander thanks to che220!
 * Support for 2018 Lexus NX 300h thanks to kengggg!
 * Speed up ECU firmware query
 * Fix bug where manager would sometimes hang after shutting down the car

Version 0.7.2 (2020-02-07)
========================
 * ECU firmware version based fingerprinting for Honda & Toyota
 * New driving model: improved path prediction during turns and lane changes and better lead speed tracking
 * Improve driver monitoring under extreme lighting and add low accuracy alert
 * Support for 2019 Rav4 Hybrid thanks to illumiN8i!
 * Support for 2016, 2017 and 2020 Lexus RX thanks to illumiN8i!
 * Support for 2020 Chrysler Pacifica Hybrid thanks to adhintz!

Version 0.7.1 (2020-01-20)
========================
 * comma two support!
 * Lane Change Assist above 45 mph!
 * Replace zmq with custom messaging library, msgq!
 * Supercombo model: calibration and driving models are combined for better lead estimate
 * More robust updater thanks to jyoung8607! Requires NEOS update
 * Improve low speed ACC tuning

Version 0.7 (2019-12-13)
========================
 * Move to SCons build system!
 * Add Lane Departure Warning (LDW) for all supported vehicles!
 * NEOS update: increase wifi speed thanks to jyoung8607!
 * Adaptive driver monitoring based on scene
 * New driving model trained end-to-end: improve lane lines and lead detection
 * Smarter torque limit alerts for all cars
 * Improve GM longitudinal control: proper computations for 15Hz radar
 * Move GM port, Toyota with DSU removed, comma pedal in community features; toggle switch required
 * Remove upload over cellular toggle: only upload qlog and qcamera files if not on wifi
 * Refactor Panda code towards ISO26262 and SIL2 compliancy
 * Forward stock FCW for Honda Nidec
 * Volkswagen port now standard: comma Harness intercepts stock camera

Version 0.6.6 (2019-11-05)
========================
 * Volkswagen support thanks to jyoung8607!
 * Toyota Corolla Hybrid with TSS 2.0 support thanks to u8511049!
 * Lexus ES with TSS 2.0 support thanks to energee!
 * Fix GM ignition detection and lock safety mode not required anymore
 * Log panda firmware and dongle ID thanks to martinl!
 * New driving model: improve path prediction and lead detection
 * New driver monitoring model, 4x smaller and running on DSP
 * Display an alert and don't start openpilot if panda has wrong firmware
 * Fix bug preventing EON from terminating processes after a drive
 * Remove support for Toyota giraffe without the 120Ohm resistor

Version 0.6.5 (2019-10-07)
========================
 * NEOS update: upgrade to Python3 and new installer!
 * comma Harness support!
 * New driving model: improve path prediction
 * New driver monitoring model: more accurate face and eye detection
 * Redesign offroad screen to display updates and alerts
 * Increase maximum allowed acceleration
 * Prevent car 12V battery drain by cutting off EON charge after 3 days of no drive
 * Lexus CT Hybrid support thanks to thomaspich!
 * Louder chime for critical alerts
 * Add toggle to switch to dashcam mode
 * Fix "invalid vehicle params" error on DSU-less Toyota

Version 0.6.4 (2019-09-08)
========================
 * Forward stock AEB for Honda Nidec
 * Improve lane centering on banked roads
 * Always-on forward collision warning
 * Always-on driver monitoring, except for right hand drive countries
 * Driver monitoring learns the user's normal driving position
 * Honda Fit support thanks to energee!
 * Lexus IS support

Version 0.6.3 (2019-08-12)
========================
 * Alert sounds from EON: requires NEOS update
 * Improve driver monitoring: eye tracking and improved awareness logic
 * Improve path prediction with new driving model
 * Improve lane positioning with wide lanes and exits
 * Improve lateral control on RAV4
 * Slow down for turns using model
 * Open sourced regression test to verify outputs against reference logs
 * Open sourced regression test to sanity check all car models

Version 0.6.2 (2019-07-29)
========================
 * New driving model!
 * Improve lane tracking with double lines
 * Strongly improve stationary vehicle detection
 * Strongly reduce cases of braking due to false leads
 * Better lead tracking around turns
 * Improve cut-in prediction by using neural network
 * Improve lateral control on Toyota Camry and C-HR thanks to zorrobyte!
 * Fix unintended openpilot disengagements on Jeep thanks to adhintz!
 * Fix delayed transition to offroad when car is turned off

Version 0.6.1 (2019-07-21)
========================
 * Remote SSH with comma prime and [ssh.comma.ai](https://ssh.comma.ai)
 * Panda code Misra-c2012 compliance, tested against cppcheck coverage
 * Lockout openpilot after 3 terminal alerts for driver distracted or unresponsive
 * Toyota Sienna support thanks to wocsor!

Version 0.6 (2019-07-01)
========================
 * New model, with double the pixels and ten times the temporal context!
 * Car should not take exits when in the right lane
 * openpilot uses only ~65% of the CPU (down from 75%)
 * Routes visible in connect/explorer after only 0.2% is uploaded (qlogs)
 * loggerd and sensord are open source, every line of openpilot is now open
 * Panda safety code is MISRA compliant and ships with a signed version on release2
 * New NEOS is 500MB smaller and has a reproducible usr/pipenv
 * Lexus ES Hybrid support thanks to wocsor!
 * Improve tuning for supported Toyota with TSS 2.0
 * Various other stability improvements

Version 0.5.13 (2019-05-31)
==========================
 * Reduce panda power consumption by 70%, down to 80mW, when car is off (not for GM)
 * Reduce EON power consumption by 40%, down to 1100mW, when car is off
 * Reduce CPU utilization by 20% and improve stability
 * Temporarily remove mapd functionalities to improve stability
 * Add openpilot record-only mode for unsupported cars
 * Synchronize controlsd to boardd to reduce latency
 * Remove panda support for Subaru giraffe

Version 0.5.12 (2019-05-16)
==========================
 * Improve lateral control for the Prius and Prius Prime
 * Compress logs before writing to disk
 * Remove old driving data when storage reaches 90% full
 * Fix small offset in following distance
 * Various small CPU optimizations
 * Improve offroad power consumption: require NEOS Update
 * Add default speed limits for Estonia thanks to martinl!
 * Subaru Crosstrek support thanks to martinl!
 * Toyota Avalon support thanks to njbrown09!
 * Toyota Rav4 with TSS 2.0 support thanks to wocsor!
 * Toyota Corolla with TSS 2.0 support thanks to wocsor!

Version 0.5.11 (2019-04-17)
========================
 * Add support for Subaru
 * Reduce panda power consumption by 60% when car is off
 * Fix controlsd lag every 6 minutes. This would sometimes cause disengagements
 * Fix bug in controls with new angle-offset learner in MPC
 * Reduce cpu consumption of ubloxd by rewriting it in C++
 * Improve driver monitoring model and face detection
 * Improve performance of visiond and ui
 * Honda Passport 2019 support
 * Lexus RX Hybrid 2019 support thanks to schomems!
 * Improve road selection heuristic in mapd
 * Add Lane Departure Warning to dashboard for Toyota thanks to arne182

Version 0.5.10 (2019-03-19)
========================
 * Self-tuning vehicle parameters: steering offset, tire stiffness and steering ratio
 * Improve longitudinal control at low speed when lead vehicle harshly decelerates
 * Fix panda bug going unexpectedly in DCP mode when EON is connected
 * Reduce white panda power consumption by 500mW when EON is disconnected by turning off WIFI
 * New Driver Monitoring Model
 * Support QR codes for login using comma connect
 * Refactor comma pedal FW and use CRC-8 checksum algorithm for safety. Reflashing pedal is required.
   Please see `#hw-pedal` on [discord](discord.comma.ai) for assistance updating comma pedal.
 * Additional speed limit rules for Germany thanks to arne182
 * Allow negative speed limit offsets

Version 0.5.9 (2019-02-10)
========================
 * Improve calibration using a dedicated neural network
 * Abstract planner in its own process to remove lags in controls process
 * Improve speed limits with country/region defaults by road type
 * Reduce mapd data usage with gzip thanks to eFiniLan
 * Zip log files in the background to reduce disk usage
 * Kia Optima support thanks to emmertex!
 * Buick Regal 2018 support thanks to HOYS!
 * Comma pedal support for Toyota thanks to wocsor! Note: tuning needed and not maintained by comma
 * Chrysler Pacifica and Jeep Grand Cherokee support thanks to adhintz!

Version 0.5.8 (2019-01-17)
========================
 * Open sourced visiond
 * Auto-slowdown for upcoming turns
 * Chrysler/Jeep/Fiat support thanks to adhintz!
 * Honda Civic 2019 support thanks to csouers!
 * Improve use of car display in Toyota thanks to arne182!
 * No data upload when connected to Android or iOS hotspots and "Enable Upload Over Cellular" setting is off
 * EON stops charging when 12V battery drops below 11.8V

Version 0.5.7 (2018-12-06)
========================
 * Speed limit from OpenStreetMap added to UI
 * Highlight speed limit when speed exceeds road speed limit plus a delta
 * Option to limit openpilot max speed to road speed limit plus a delta
 * Cadillac ATS support thanks to vntarasov!
 * GMC Acadia support thanks to CryptoKylan!
 * Decrease GPU power consumption
 * NEOSv8 autoupdate

Version 0.5.6 (2018-11-16)
========================
 * Refresh settings layout and add feature descriptions
 * In Honda, keep stock camera on for logging and extra stock features; new openpilot giraffe setting is 0111!
 * In Toyota, option to keep stock camera on for logging and extra stock features (e.g. AHB); 120Ohm resistor required on giraffe.
 * Improve camera calibration stability
 * More tuning to Honda positive accelerations
 * Reduce brake pump use on Hondas
 * Chevrolet Malibu support thanks to tylergets!
 * Holden Astra support thanks to AlexHill!

Version 0.5.5 (2018-10-20)
========================
 * Increase allowed Honda positive accelerations
 * Fix sporadic unexpected braking when passing semi-trucks in Toyota
 * Fix gear reading bug in Hyundai Elantra thanks to emmertex!

Version 0.5.4 (2018-09-25)
========================
 * New Driving Model
 * New Driver Monitoring Model
 * Improve longitudinal mpc in mid-low speed braking
 * Honda Accord hybrid support thanks to energee!
 * Ship mpc binaries and sensibly reduce build time
 * Calibration more stable
 * More Hyundai and Kia cars supported thanks to emmertex!
 * Various GM Volt improvements thanks to vntarasov!

Version 0.5.3 (2018-09-03)
========================
 * Hyundai Santa Fe support!
 * Honda Pilot 2019 support thanks to energee!
 * Toyota Highlander support thanks to daehahn!
 * Improve steering tuning for Honda Odyssey

Version 0.5.2 (2018-08-16)
========================
 * New calibration: more accurate, a lot faster, open source!
 * Enable orbd
 * Add little endian support to CAN packer
 * Fix fingerprint for Honda Accord 1.5T
 * Improve driver monitoring model

Version 0.5.1 (2018-08-01)
========================
 * Fix radar error on Civic sedan 2018
 * Improve thermal management logic
 * Alpha Toyota C-HR and Camry support!
 * Auto-switch Driver Monitoring to 3 min counter when inaccurate

Version 0.5 (2018-07-11)
========================
 * Driver Monitoring (beta) option in settings!
 * Make visiond, loggerd and UI use less resources
 * 60 FPS UI
 * Better car parameters for most cars
 * New sidebar with stats
 * Remove Waze and Spotify to free up system resources
 * Remove rear view mirror option
 * Calibration 3x faster

Version 0.4.7.2 (2018-06-25)
==========================
 * Fix loggerd lag issue
 * No longer prompt for updates
 * Mitigate right lane hugging for properly mounted EON (procedure on wiki)

Version 0.4.7.1 (2018-06-18)
==========================
 * Fix Acura ILX steer faults
 * Fix bug in mock car

Version 0.4.7 (2018-06-15)
==========================
 * New model!
 * GM Volt (and CT6 lateral) support!
 * Honda Bosch lateral support!
 * Improve actuator modeling to reduce lateral wobble
 * Minor refactor of car abstraction layer
 * Hack around orbd startup issue

Version 0.4.6 (2018-05-18)
==========================
 * NEOSv6 required! Will autoupdate
 * Stability improvements
 * Fix all memory leaks
 * Update C++ compiler to clang6
 * Improve front camera exposure

Version 0.4.5 (2018-04-27)
==========================
 * Release notes added to the update popup
 * Improve auto shut-off logic to disallow empty battery
 * Added onboarding instructions
 * Include orbd, the first piece of new calibration algorithm
 * Show remaining upload data instead of file numbers
 * Fix UI bugs
 * Fix memory leaks

Version 0.4.4 (2018-04-13)
==========================
 * EON are flipped! Flip your EON's mount!
 * Alpha Honda Ridgeline support thanks to energee!
 * Support optional front camera recording
 * Upload over cellular toggle now applies to all files, not just video
 * Increase acceleration when closing lead gap
 * User now prompted for future updates
 * NEO no longer supported :(

Version 0.4.3.2 (2018-03-29)
============================
 * Improve autofocus
 * Improve driving when only one lane line is detected
 * Added fingerprint for Toyota Corolla LE
 * Fixed Toyota Corolla steer error
 * Full-screen driving UI
 * Improved path drawing

Version 0.4.3.1 (2018-03-19)
============================
 * Improve autofocus
 * Add check for MPC solution error
 * Make first distracted warning visual only

Version 0.4.3 (2018-03-13)
==========================
 * Add HDR and autofocus
 * Update UI aesthetic
 * Grey panda works in Waze
 * Add alpha support for 2017 Honda Pilot
 * Slight increase in acceleration response from stop
 * Switch CAN sending to use CANPacker
 * Fix pulsing acceleration regression on Honda
 * Fix openpilot bugs when stock system is in use
 * Change starting logic for chffrplus to use battery voltage

Version 0.4.2 (2018-02-05)
==========================
 * Add alpha support for 2017 Lexus RX Hybrid
 * Add alpha support for 2018 ACURA RDX
 * Updated fingerprint to include Toyota Rav4 SE and Prius Prime
 * Bugfixes for Acura ILX and Honda Odyssey

Version 0.4.1 (2018-01-30)
==========================
 * Add alpha support for 2017 Toyota Corolla
 * Add alpha support for 2018 Honda Odyssey with Honda Sensing
 * Add alpha support for Grey Panda
 * Refactored car abstraction layer to make car ports easier
 * Increased steering torque limit on Honda CR-V by 30%

Version 0.4.0.2 (2018-01-18)
==========================
 * Add focus adjustment slider
 * Minor bugfixes

Version 0.4.0.1 (2017-12-21)
==========================
 * New UI to match chffrplus
 * Improved lateral control tuning to fix oscillations on Civic
 * Add alpha support for 2017 Toyota Rav4 Hybrid
 * Reduced CPU usage
 * Removed unnecessary utilization of fan at max speed
 * Minor bug fixes

Version 0.3.9 (2017-11-21)
==========================
 * Add alpha support for 2017 Toyota Prius
 * Improved longitudinal control using model predictive control
 * Enable Forward Collision Warning
 * Acura ILX now maintains openpilot engaged at standstill when brakes are applied

Version 0.3.8.2 (2017-10-30)
==========================
 * Add alpha support for 2017 Toyota RAV4
 * Smoother lateral control
 * Stay silent if stock system is connected through giraffe
 * Minor bug fixes

Version 0.3.7 (2017-09-30)
==========================
 * Improved lateral control using model predictive control
 * Improved lane centering
 * Improved GPS
 * Reduced tendency of path deviation near right side exits
 * Enable engagement while the accelerator pedal is pressed
 * Enable engagement while the brake pedal is pressed, when stationary and with lead vehicle within 5m
 * Disable engagement when park brake or brake hold are active
 * Fixed sporadic longitudinal pulsing in Civic
 * Cleanups to vehicle interface

Version 0.3.6.1 (2017-08-15)
============================
 * Mitigate low speed steering oscillations on some vehicles
 * Include board steering check for CR-V

Version 0.3.6 (2017-08-08)
==========================
 * Fix alpha CR-V support
 * Improved GPS
 * Fix display of target speed not always matching HUD
 * Increased acceleration after stop
 * Mitigated some vehicles driving too close to the right line

Version 0.3.5 (2017-07-30)
==========================
 * Fix bug where new devices would not begin calibration
 * Minor robustness improvements

Version 0.3.4 (2017-07-28)
==========================
 * Improved model trained on more data
 * Much improved controls tuning
 * Performance improvements
 * Bugfixes and improvements to calibration
 * Driving log can play back video
 * Acura only: system now stays engaged below 25mph as long as brakes are applied

Version 0.3.3  (2017-06-28)
===========================
 * Improved model trained on more data
 * Alpha CR-V support thanks to energee and johnnwvs!
 * Using the opendbc project for DBC files
 * Minor performance improvements
 * UI update thanks to pjlao307
 * Power off button
 * 6% more torque on the Civic

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
