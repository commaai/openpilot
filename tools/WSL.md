# Openpilot on WSL2 Ubuntu 24.04

[Windows Subsystem for Linux 2 (WSL2)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Ubuntu apart from some known issues enumarted here:

## Known Issues

### Webcam support

[Issues](https://github.com/commaai/openpilot/issues/34216) have been reported with developers attempting to run Openpilot with webcam support.  Successfully achieving openpilot webcam support in WSL2 will likely involve [attaching USB devices to WSL](https://learn.microsoft.com/en-us/windows/wsl/connect-usb) and [rebuilding the Kernel](https://www.youtube.com/watch?v=t_YnACEPmrM).

### GUI Issues

If you are running WSL and any GUIs are failing (segfaulting or other strange issues) even after [installing a GPU driver](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps#prerequisites), you may need to enable software rendering with `LIBGL_ALWAYS_SOFTWARE=1`, e.g. `LIBGL_ALWAYS_SOFTWARE=1 selfdrive/ui/ui`.

### IPC outside WSL2

Because of WSL2's virtual IP address configuration, interprocess communication with processes running on a remote host will have issues communicating with processes running in WSL2.

### OpenCL support

Bro, I STILL haven't figured this one out! `sudo apt install pocl-opencl-icd`?

## WSL Setup Instructions

- [WSL2 initial setup](https://docs.microsoft.com/en-us/windows/wsl/install)
- [Ubunt 24.04 installer from Microsoft Store](https://apps.microsoft.com/detail/9NZ3KLHXDJP5?hl=en-us&gl=CA&ocid=pdpshare)
- [Install GPU driver for better GUI performance and reliability](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps#prerequisites)
