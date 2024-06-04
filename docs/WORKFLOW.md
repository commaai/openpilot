# openpilot development workflow

Aside from the ML models, most tools used for openpilot development are in this repo.

Most development happens on normal Ubuntu workstations, and not in cars or directly on comma devices. See the [setup guide](../tools) for getting your PC setup for openpilot development.

## System Requirements

openpilot is developed and tested on **Ubuntu 20.04**, which is the primary development target aside from the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car).

Running natively on any other system is not recommended and will require modifications. On Windows you can use WSL, and on macOS or incompatible Linux systems, it is recommended to use the dev containers.

## Native setup on Ubuntu 20.04

```bash
# clone openpilot
git clone --recurse-submodules https://github.com/commaai/openpilot.git

# get latest commits from master
git pull
git submodule update --init --recursive
```

For quick setup, devs can run `ubuntu_setup.sh` which will go through dependency installs for both `ubuntu` and `python` while also setting up a `venv` to use right away. However if you'd like to pick dependencies based on groups you can run `install_ubuntu_dependencies.sh` separately and `install_python_dependencies.sh` in your own virtual env.

```bash
# update dependencies
tools/ubuntu_setup.sh

# activate a venv shell with the Python dependencies installed:
poetry shell

# build everything
scons -j$(nproc)

# build just the ui with either of these
scons -j8 selfdrive/ui/
cd selfdrive/ui/ && scons -u -j8

# test everything
pytest .

# test just logging services
cd system/loggerd && pytest .

# run the linter
pre-commit run --all
```

## Dev Container on any Linux or macOS

openpilot supports [Dev Containers](https://containers.dev/). Dev containers provide customizable and consistent development environment wrapped inside a container. This means you can develop in a designated environment matching our primary development target, regardless of your local setup.

Dev containers are supported in [multiple editors and IDEs](https://containers.dev/supporting), including Visual Studio Code. Use the following [guide](https://code.visualstudio.com/docs/devcontainers/containers) to start using them with VSCode.

#### X11 forwarding on macOS

GUI apps like `ui` or `cabana` can also run inside the container by leveraging X11 forwarding. To make use of it on macOS, additional configuration steps must be taken. Follow [these](https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088) steps to setup X11 forwarding on macOS.

## WSL on Windows

[Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Ubuntu. [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/compare-versions) specifically has been reported by several users to be a seamless experience.

Follow [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install) to setup the WSL and install the `Ubuntu-20.04` distribution. Once your Ubuntu WSL environment is setup, follow the Linux setup instructions to finish setting up your environment. See [these instructions](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for running GUI apps.

**NOTE**: If you are running WSL and any GUIs are failing (segfaulting or other strange issues) even after following the steps above, you may need to enable software rendering with `LIBGL_ALWAYS_SOFTWARE=1`, e.g. `LIBGL_ALWAYS_SOFTWARE=1 selfdrive/ui/ui`.

## Testing

### Automated Testing

All PRs and commits are automatically checked by GitHub Actions. Check out `.github/workflows/` for what GitHub Actions runs. Any new tests should be added to GitHub Actions.

### Code Style and Linting

Code is automatically checked for style by GitHub Actions as part of the automated tests. You can also run these tests yourself by running `pre-commit run --all`.
