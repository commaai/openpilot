![](https://i.ibb.co/09tj5Q6/all.jpg)

FlexRay adapter for [openpilot](http://github.com/commaai/openpilot)
======

Features
------

- FlexRay driver for MPC5748G
- FlexRay adapter for [EON Dashcam DevKit](https://comma.ai/shop/products/EON-dashcam-devkit) and PC (Windows 7/10, Ubuntu), send & receive FlexRay frames over ethernet(TCP).
- Flexray boardd implementation for OpenPilot, forward FlexRay frames to ZMQ.
- A GUI application running on PC for facilitating FlexRay configuration.
    - create/edit/load/export FlexRay configuration file.
    - Validate/Auto-calculate FlexRay parameters according to the constraints defined in FlexRay spec 2.1 appendix A.
    - Troubleshoot FlexRay configuration problems, view protocol status data on the board.
    - Start/stop FlexRay driver.
    - Change the FlexRay configuration on the adapter.
    - Sniff on FlexRay bus, display received frames.
    - Send frames, customize frame ID and payload content.

Hardware Support
------

Right now the FlexRay adapter supports the MPC5748G chip, tested on [DEVKIT MPC5748G](https://www.nxp.com/products/processors-and-microcontrollers/power-architecture-processors/mpc5xxx-55xx-32-bit-mcus/ultra-reliable-mpc57xx-32-bit-automotive-and-industrial-microcontrollers-mcus/development-board-for-mpc5748g:DEVKIT-MPC5748G).


Directory structure
------
    .
    └── flexray_adapter                 # NXP S32 Design Studio project root directory
        └── Sources
            ├── flexray_driver.c        # FlexRay driver API code
            ├── flexray_config.h        # FlexRay configuration parameters definition
            ├── flexray_state_machine.c # FlexRay state machine, control the FlexRay CC via FlexRay driver API
            ├── tcp_interface.c         # TCP interfacing implementation
            ├── tcpip_stack.c           # lwIP TCP/IP stack launching
            └── packet.h                # A packet based protocol for TCP interfacing
    └── python
        ├── flexray_config.py           # FlexRay configuration file API
        ├── tcp_interface.py            # FlexRay TCP interfacing API
        ├── gui.py                      # FlexRay test & configuration tool
        └── test/                       # Sample FlexRay configuration YAML files
    ..
    └── selfdrive
            └── flexrayd
                ├── flexrayd.c          # FlexRay boardd
                └── flexrayd_test.py    # flexrayd test program

How to compile
------
- OS requirements
    - Windows 10
    - Linux may also be OK but never get tested.
- Download and install [S32 Design Studio for Power Architecture 2017.R1 ](https://www.nxp.com/support/developer-resources/run-time-software/s32-design-studio-ide/s32-design-studio-ide-for-power-architecture-based-mcus:S32DS-PA?tab=Design_Tools_Tab) to default position( %SystemDrive%\NXP). A free account registration will be required.
- Install S32DS for Power Architecture 2017.R1 Update 7, make sure the MPC574xx SDK RTM 1.0.0 is installed:
  1. On the S32DS download page, find the Update 7, download and unzip to local folder.  
  2. Launch S32 design studio, open "Help" menu, choose "Install new software...", then click "Add..." button to open Add Repository dialog, click "Local..." button, browse to the folder where Update 7 is unzipped, click "OK" button. 
  3. Make sure "MPC74xx SDK RTM 1.0.0" is checked, then begin installation.
  ![]( https://i.ibb.co/C9JMbYr/SDK-RTM-1-0-0.png)
- Create a workspace in S32DS, import flexray_adapter project into current workspace.
- Build the project.

How to debug
------
- Connect DEVKIT MPC5748G to computer via the USB OpenSDA port.
- Choose DEBUG_RAM configuration and build.
- Set the debugger to OpenSDA in debug configuration window.
- Start debugging.

How to test
------

- ### Establish FlexRay communication between PC and EON
    - Hardware preparation
        - A PC running Windows 10 or Ubuntu Desktop.
        - A EON running OpenPilot.
        - A [Comma SMAYS](https://comma.ai/shop/products/comma-smays-adapter/) for EON interfacing.
        - Two MPC5748G DEVKIT boards flashed using the output ELF file from compilation of the flexray_adapter S32S project.
    - Connect all hardware together
        - Connect the MPC5748 boards via the FlexRay port using a 1.25mm 2pin connector.
        - Connect a MPC5748 board to EON via Comma SMAYS and an ethernet cable, set the IP address of ethernet adapter to 192.168.5.10.
        - Connect the USB port of Comma SMAYS to PC for power supply.
        - Connect the other MPC5748 board to PC via an ethernet cable, set the IP address of ethernet adapter to 192.168.5.10.
    - Start FlexRay communication On EON
        - Start test python script and flexrayd:
        ```bash
        root@localhost:/data/openpilot/selfdrive/flexrayd$ ./flexrayd_test.py
        Loading FlexRay config file: ../../flexray_adapter/python/test/node1.yml...
        Tx Static Slot ids:  [1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
        Tx Dynamic Slot ids:  [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
        Listening for FlexRay frames
         Type a tx slot id to send a frame, 'q' to exit, 'h' to show tx slot ids:
        ```
        ```bash
        root@localhost:/data/openpilot/selfdrive/flexrayd$ ./flexrayd
        flexrayd.cc: starting flexrayd
        connecting
        flexrayd.cc: waiting for flexray params
        flexrayd.cc: got 892 bytes of flexray params
        ```
    - Start FlexRay communication on PC
        - Launch flexray_tool.py.
        - Click "Join into FlexRay Network" button.
        - From the dropdown list, choose node2.yml, click "Load selected file" button.
        ![](https://i.ibb.co/wBjSVtS/Connect-To-Flex-Ray.png)
        - Click "Connect" button
    - After successful connection, flexray_tool GUI will show "POC status: normal active", and the number of sync frames will become 4.
    ![](https://i.ibb.co/841NtCb/Flex-Ray-Tool.png)
    - Send frame test
        - Send frame on static slot id 1 on EON
            ```bash
            root@localhost:/data/openpilot/selfdrive/flexrayd$ ./flexrayd_test.py
            Loading FlexRay config file: ../../flexray_adapter/python/test/node1.yml...
            Tx Static Slot ids:  [1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,                             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,                             60]
            Tx Dynamic Slot ids:  [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 7                            5, 76, 77, 78, 79, 80]
            Listening for FlexRay frames
             Type a tx slot id to send a frame, 'q' to exit, 'h' to show tx slot ids:1
            Start sending on slot id 1...
            Type a tx slot id to send a frame, 'q' to exit, 'h' to show tx slot ids:
            ```
            Inspect the frames on PC
            ![](https://i.ibb.co/sJ80RRb/RxFrames.png)
        - Send frame on static slot id 2 on PC
            - Click "Send frame" button.
            - Click the Checkbox before "static slot 2", fill the payload with some data, then click "Start sending" button.
            ![](https://i.ibb.co/2jmMrBG/Send-Frame.png)
            Inspect the frames on EON
            ```bash
            Frame 0 , ID: 2 , Payload Len: 64 ['0x55', '0x55', '0x55', '0x55', '0x55', '0x55', '0x55', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0', '0x0']
            ```

Some thoughts on how to connect the adapter to car's FlexRay network 
------
- Capture the car's FlexRay traffic
    - Find the FlexRay port where the LKAS camera is connected.
    - Connect a oscilloscope(like [PicoScope](https://www.picotech.com/products/oscilloscope)) to the FlexRay BP or BM port.
    - Start the car, capture a waveform.
- Estimate the FlexRay parameters
    - Decode the FlexRay frame from the waveform using PicoScope software to get slot/frame parameters, such as gdPayloadLengthStatic.
    - Analyze the waveform to estimate timing parameters, such as gdTSSTransmitter, gdMacroTick, gdCycle, etc.
- Connect test
    - Use the estimated parameters to create a config file in flexray_tool.
    - Connect FlexRay adapter to the car instead of the oscilloscope.
    - Use the config file to join the FlexRay network.

 Reference Documents
------
- [MPC5748G Reference Manual](https://www.nxp.com/docs/en/reference-manual/MPC5748GRM.pdf)
- [DEVKIT-MPC5478G Quick Start Guide](https://www.nxp.com/docs/en/quick-reference-guide/DEVKIT-MPC5748G-QSG.pdf)
- FlexRay Protocol Specification Version 2.1 Revision A

Licensing
------

MIT license. 
