
openpilot is developed and tested on **Ubuntu 20.04**, which is the primary development target aside from the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car).

Running natively on any other system is not recommended and will require modifications. On Windows you can use WSL, and on macOS or incompatible Linux systems, it is recommended to use the dev containers.
Running natively on any other system is not recommended and will require modifications. On Windows you can use Windows Subsystem for Linux (WSL), and on macOS or incompatible Linux systems, it is recommended to use the Dev Containers.

## WSL on Windows

[Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Ubuntu. [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/compare-versions) specifically has been reported by several users to be a seamless experience.

Follow [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install) to setup the WSL and install the `Ubuntu-20.04` distribution. Once your Ubuntu WSL environment is setup, follow the Linux setup instructions to finish setting up your environment. See [these instructions](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for running GUI apps.
Follow the below instructions to setup WSL:
1) Open PowerShell or Windows Command Prompt (run as administrator) and enter the command without quotes: "wsl -- install"
2) Install the Ubuntu-20.04 WSL distribution if required using the command without quotes: "wsl --install -d `Ubuntu-20.04` distribution" 

Setting up your environment:
1) Open PowerShell or Windows Command Prompt (run as administrator) and enter the command without quotes: "wsl -- update"
2) Enter the command "wsl --shutdown"
3) Enter the command "sudo apt update"
4) Enter the command "sudo apt install gnome-text-editor -y" to install the Gnome Text Editor

If you would like to see more details instructions and explanations of commands, see the following links:
- Full WSL Installation instructions: https://docs.microsoft.com/en-us/windows/wsl/install)
- Full WSL GUI Apps Installation instructions: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps

**NOTE**: If you are running WSL and any GUIs are failing (segfaulting or other strange issues) even after following the steps above, you may need to enable software rendering with `LIBGL_ALWAYS_SOFTWARE=1`, e.g. `LIBGL_ALWAYS_SOFTWARE=1 selfdrive/ui/ui`.


# Dev Container on any Linux or macOS

openpilot supports [Dev Containers](https://containers.dev/). Dev containers provide customizable and consistent development environment wrapped inside a container. This means you can develop in a designated environment matching our primary development target, regardless of your local setup.

Dev containers are supported in [multiple editors and IDEs](https://containers.dev/supporting), including Visual Studio Code. Use the following [guide](https://code.visualstudio.com/docs/devcontainers/containers) to start using them with VSCode.

**NOTE**: If you are running WSL and any GUIs are failing (segfaulting or other strange issues) even after following the steps above, you may need to enable software rendering with `LIBGL_ALWAYS_SOFTWARE=1`, e.g. `LIBGL_ALWAYS_SOFTWARE=1 selfdrive/ui/ui`.


## Native setup on Ubuntu 20.04

@@ -41,24 +71,10 @@ poetry shell
scons -u -j$(nproc)
```

#### X11 forwarding on macOS

GUI apps like `ui` or `cabana` can also run inside the container by leveraging X11 forwarding. To make use of it on macOS, additional configuration steps must be taken. Follow [these](https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088) steps to setup X11 forwarding on macOS.

## CTF
Learn about the openpilot ecosystem and tools by playing our [CTF](/tools/CTF.md).
