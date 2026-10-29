# openpilot tools

## System Requirements

openpilot supports Linux alongside the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car).

Most of openpilot should work natively on macOS. On Windows, you can use WSL for a nearly native Linux experience.

## Native setup on Linux and macOS

Follow these instructions for a fully managed setup experience. If you'd like to manage the dependencies yourself, just read the setup scripts in this directory.

**1. Clone openpilot**
``` bash
git clone https://github.com/commaai/openpilot.git
```

**2. Run the setup script**
``` bash
cd openpilot
tools/op.sh setup
```

**3. Activate a Python shell**
Activate a shell with the Python dependencies installed:
``` bash
source .venv/bin/activate
```

**4. Build openpilot**
``` bash
scons -u
```

## Large files

Git LFS stores large files in the Hugging Face bucket `commaai/openpilot-lfs`,
under `sha256/<object hash>`. `op setup` installs the bucket transfer agent;
normal Git add, checkout, pull, and push continue to use Git LFS. Existing
commands such as `git lfs pull`, `fetch`, `checkout`, and `prune` still work.

For a manual setup, clone with `GIT_LFS_SKIP_SMUDGE=1`, run `git lfs install --local`
and `./scripts/lfs_transfer.py install`, then `git lfs pull`. The installed agent
lives in Git's common directory so it survives branch switches and works in
linked worktrees. Device startup and release builds also configure the agent.

Downloads work with plain Python over HTTP, or use Xet when its dependencies
are installed. Set `HF_HUB_DISABLE_XET=1` to use HTTP. Uploads require `uv` on
`PATH`; the agent loads its pinned Hugging Face and Xet dependencies separately
from openpilot's environment. Use `hf auth login` or set `HF_TOKEN` to a token
with write access to the bucket. Public downloads do not require authentication.

Set `OPENPILOT_LFS_BUCKET=owner/bucket` to use another bucket. It must already
exist and contain the objects referenced by the branches you use. Fork objects
must be copied to the main bucket before merging. Transfers use the bucket
directly; the legacy LFS server is not used.

## WSL on Windows

[Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Linux. [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/compare-versions) specifically has been reported by several users to be a seamless experience.

Follow [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install) to set up WSL. We recommend installing the `Ubuntu-24.04` distribution. Once your WSL environment is set up, follow the Linux setup instructions to finish setting up your environment. See [these instructions](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for running GUI apps.

**NOTE**: If you are running WSL 2 and experiencing performance issues with the UI or simulator, you may need to explicitly enable hardware acceleration by setting `GALLIUM_DRIVER=d3d12` before commands. Add `export GALLIUM_DRIVER=d3d12` to your `~/.bashrc` file to make it automatic for future sessions.

## CTF
Learn about the openpilot ecosystem and tools by playing our [CTF](/tools/CTF.md).

## Directory Structure

```
├── car_porting/        # Tools for porting new cars
├── release/            # Scripts for building openpilot releases
└── scripts/            # Miscellaneous scripts
```

Development tools such as cabana, plotjuggler, and replay live in [openpilot/tools/](/openpilot/tools/):

```
├── cabana/             # View and plot CAN messages from drives or in realtime
├── camerastream/       # Cameras stream over the network
├── joystick/           # Control your car with a joystick
├── lib/                # Libraries to support the tools and reading openpilot logs
├── plotjuggler/        # A tool to plot openpilot logs
├── replay/             # Replay drives and mock openpilot services
└── sim/                # Run openpilot in a simulator
```
