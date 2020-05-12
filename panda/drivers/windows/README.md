```
                                                                                                        ;"   ^;     ;'   ",
______/\\\\\\\\\\\____/\\\\\\\\\_______/\\\\\\\\\\\\\\\______/\\\\\\\\\\_____________/\\\____           ;    s$$$$$$$s     ;
 _____\/////\\\///___/\\\///////\\\____\/\\\///////////_____/\\\///////\\\__________/\\\\\____          ,  ss$$$$$$$$$$s  ,'
  _________\/\\\_____\///______\//\\\___\/\\\_______________\///______/\\\_________/\\\/\\\____         ;s$$$$$$$$$$$$$$$
   _________\/\\\_______________/\\\/____\/\\\\\\\\\\\\_____________/\\\//________/\\\/\/\\\____        $$$$$$$$$$$$$$$$$$
    _________\/\\\____________/\\\//______\////////////\\\__________\////\\\_____/\\\/__\/\\\____      $$$$P""Y$$$Y""W$$$$$
     _________\/\\\_________/\\\//____________________\//\\\____________\//\\\__/\\\\\\\\\\\\\\\\_     $$$$  p"$$$"q  $$$$$
      __/\\\___\/\\\_______/\\\/____________/\\\________\/\\\___/\\\______/\\\__\///////////\\\//__    $$$$  .$$$$$.  $$$$
       _\//\\\\\\\\\_______/\\\\\\\\\\\\\\\_\//\\\\\\\\\\\\\/___\///\\\\\\\\\/_____________\/\\\____  _ $$$$$$$$$$$$$$$$
        __\/////////_______\///////////////___\/////////////_______\/////////_______________\///_____| |  "Y$$$"*"$$$Y"
                                                                                _ __   __ _ _ __   __| | __ _"$b.$$"
                                                                               | '_ \ / _` | '_ \ / _` |/ _` |
                                                                               | |_) | (_| | | | | (_| | (_| |
                                                                               | .__/ \__,_|_| |_|\__,_|\__,_|
                                                                               | |     A comma.ai product.
                                                                               |_| (Code by Jessy Diamond Exum)
```


# Installing J2534 driver:

[Download](https://github.com/commaai/panda/files/4017364/panda.J2534.driver.install.zip)

Depending on what version of windows you are on, you may need to separately install the WinUSB driver (see next section).

# Installing WinUSB driver:

Installation automatically happens for Windows 8 and Windows 10 because the panda
firmware contains the USB descriptors necessary to auto-install the WinUSB driver.

Windows 7 will not auto-install the WinUSB driver. You can use Zadig to install
the WinUSB driver. This software is not tested on anything before 7.

More details here:
[WinUSB (Winusb.sys) Installation](https://docs.microsoft.com/en-us/windows-hardware/drivers/usbcon/winusb-installation)
[WCID Devices](https://github.com/pbatard/libwdi/wiki/WCID-Devices)
[Zadig for installing libusb compatible driver](https://github.com/pbatard/libwdi/wiki/Zadig)

# Using J2534:

After installing the J2534 drivers for the panda, you can do... nothing.
You first need to get a J2534 client that can load the drivers and talk to
the panda for you.

A simple tool for testing J2534 drivers is DrewTech's 'J2534-1 Bus Analysis
Tool' available in the 'Other Support Applications' section of their
[Download Page](http://www.drewtech.com/downloads/).

# What is J2534?

J2534 is an API that tries to provide a consistent way to send/receive
messages over the many different protocols supported by the OBD II
port. The place this is perhaps most obvious, is sending data over
different protocols (each using unique packetizing methods) using the
same data format.

For each PassThru Device that should be used with J2534 (in this case,
the panda), a 'driver' has to be written that can be loaded by a
client application wanting to send/receive data.

A lot of J2534 has good ideas behind it, but the standard has some odd choices:

* Platform Locked: Requires using the Windows Registry to find installed J2534 libraries/drivers. Drivers have to be DLLs.
* Architecture Locked: So far there is only support for x86.
* No device autodetect, and poor support for selecting from multiple devices.
* Constant vague language about important behavior (small differences between vendors).
* Most common differences become standard in later revisions.

# Why use J2534 with the panda?

J2534 is the only interface supported by most professional grade
vehicle diagnostics systems (such as HDS). These tools are useful for
diagnosing vehicles, as well as reverse engineering some lesser known
features.

# What parts are supported with panda?

- [ ] **J1850VPW** *(Outdated, and not physically supported by the panda)*
- [ ] **J1850PWM** *(Outdated, and not physically supported by the panda)*
- [X] **CAN**
- [X] **ISO15765**
- [ ] **ISO9141** *(This protocol could be implemented if 5 BAUD init support is added to the panda.)*
- [ ] **ISO14230/KWP2000** *(Could be supported with FAST init, 5baud init if panda adds support for 5bps serial)*

# Building the Project:

This project is developed with Visual Studio 2017, the Windows SDK,
and the Windows Driver Kit (WDK).

The WDK is only required for creating the signed WinUSB inf file. The
WDK may also provide the headers for WinUSB.

To build all the projects required for the installer, in Visual
Studio, select **Build->Batch Build.** In the project list select:

- **"panda"** *Release|x86*
- **"panda"** *Release|x64*
- **"panda Driver Package"** Debug|x86 (Note this inf file works with x86/amd64).
- **"pandaJ2534DLL"** *Release|x86*

The installer is generated with [NullSoft NSIS](http://nsis.sourceforge.net/Main_Page).
Use NSIS to run panda_install.nsi after building all the required projects.

Before generating the installer, you must go to copy vscruntimeinfo.nsh.sample to
vscruntimeinfo.nsh and follow the instructions to bundle in the Visual Studio C
Runtime required by your version of Visual Studio. Without this runtime, the panda
code will not work, so without this file, the installer will refuse to build.

# Developing:

- Edit and merge pandaJ2534DLL\J2534register_x64.reg to register your development J2534 DLL.
- Add your output directory (panda\drivers\windows\Debug_x86) to your system PATH to avoid insanity.

# ToDo Items:

- Apply a style-guide and consistent naming convention for Classes/Functions/Variables.
- Send multiple messages (each with a different address) from a given connection at the same time.
- Implement ISO14230/KWP2000 FAST (LIN communication is already supported with the raw panda USB driver).
- Find more documentation about SW_CAN_PS (Single Wire CAN, aka GMLAN).
- Find example of client using a _PS version of a protocol (PS is pin select, and may support using different CAN buses).


# Known Issues:

- ISO15765 Multi-frame TX: Hardware delays make transmission overshoot
  STMIN by several milliseconds. This does not violate the requirements
  of STMIN, it just means it is a little slower than it could be.

- All Tx messages from a single Connection are serialized. This can be
  relaxed to allow serialization of messages based on their address
  (making multiple queues, effectively one queue per address).

# Troubleshooting:
troubleshooting:
1. Install DrewTech J2534-1 Bus Analysis Tool
http://www.drewtech.com/downloads/tools/Drew%20Technologies%20Tool%20for%20J2534-1%20API%20v1.07.msi
2. Open DrewTech tool and make sure it shows "panda" as a device listed (this means registry settings are correct)
3. When DrewTech tool attempts to load the driver it will show an error if it fails
4. To figure out why the driver fails to load install Process Monitor and filter by the appropriate process name
https://docs.microsoft.com/en-us/sysinternals/downloads/procmon

# Other:
Panda head ASCII art by dcau