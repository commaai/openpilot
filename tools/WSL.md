# Openpilot on WSL2 Ubuntu 24.04

## WSL Installation steps

- [WSL2 initial setup](https://docs.microsoft.com/en-us/windows/wsl/install)
- [Ubunt 24.04 installer from Microsoft Store](https://apps.microsoft.com/detail/9NZ3KLHXDJP5?hl=en-us&gl=CA&ocid=pdpshare)
- [CUDA drivers](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)
- [CUDNN drivers](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)

## Known Issues

[Windows Subsystem for Linux 2 (WSL2)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Ubuntu apart from some known issues enumarted here:

### Webcam support

[Issues](https://github.com/commaai/openpilot/issues/34216) have been reported with developers attempting to run Openpilot with webcam support.  Successfully achieving openpilot webcam support in WSL2 will likely involve [attaching USB devices to WSL](https://learn.microsoft.com/en-us/windows/wsl/connect-usb) and [rebuilding the Kernel](https://www.youtube.com/watch?v=t_YnACEPmrM).

### GUI Issues

If you are running WSL and any GUIs are failing (segfaulting or other strange issues) even after [installing a GPU driver](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps#prerequisites), you may need to enable software rendering with `LIBGL_ALWAYS_SOFTWARE=1`, e.g. `LIBGL_ALWAYS_SOFTWARE=1 selfdrive/ui/ui`.

### IPC outside WSL2

Because of WSL2's virtual IP address configuration, interprocess communication with processes running on a remote host will have issues communicating with processes running in WSL2.  For this reason, tools like [Camera stream](tools/camerastream) are not supported for WSL developers.
