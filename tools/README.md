# openpilot tools

## System Requirements

openpilot is developed and tested on **Ubuntu 24.04**, which is the primary development target aside from the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car).

Most of openpilot should work natively on macOS. On Windows you can use WSL for a nearly native Ubuntu experience. Running natively on any other system is not currently recommended and will likely require modifications.

## Native setup on Ubuntu 24.04 and macOS

Follow these instructions for a fully managed setup experience. If you'd like to manage the dependencies yourself, just read the setup scripts in this directory.

**1. Clone openpilot**
``` bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/commaai/openpilot.git
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

`lfs.py` stores large files in the Hugging Face bucket `commaai/openpilot-lfs`,
under `sha256/<object hash>`. `op setup` installs the Git filter and pre-push hook;
normal Git add, checkout, pull, and push then use the bucket. Run `./lfs.py pull`
to download and materialize the files in the Git index, including staged renames,
without changing staged entries or local modifications. The installed client and
cache live in Git's common directory, so older checkouts and linked worktrees keep working.
Downloads and local Git operations work with plain Python. Downloads use Xet
when installed, with HTTP as the dependency-free default. Run
`uv run --script lfs.py pull` to download with Xet, or set `HF_HUB_DISABLE_XET=1`
to use HTTP. Pushes that include large files require Xet; the hook uses uv to
load those dependencies when needed, separately from openpilot's environment.

Public downloads do not require authentication. For uploads, use `hf auth login`
or set `HF_TOKEN` to a token with write access to the bucket and the legacy
`commaai/openpilot-lfs` model repository. During the migration, new objects are
uploaded to the legacy repository before publishing them in the bucket.
`.lfsconfig` remains in source and Chestnut releases so older installers and
updaters can download them; openpilot installs the bucket client at startup.
Stock Git LFS uploads are disabled so pushes cannot publish objects only to the
legacy repository. If a push fails with `/dev/null/run-lfs.py-install`, run
`./lfs.py install` (or `tools/op.sh setup`) and retry the push.

Set `OPENPILOT_LFS_BUCKET=owner/bucket` to use another bucket. The bucket must already
exist and contain the objects referenced by the branches you need; the client
does not create buckets or delete objects. Forks need write access to their
configured bucket, and objects must be copied to the main bucket before merging.
Custom buckets do not mirror to the legacy repository by default. To support
older clients, set `OPENPILOT_LFS_LEGACY_REPO=owner/repo` and point `.lfsconfig`
`url` at that repository's LFS endpoint. Keep `pushurl` disabled.

## WSL on Windows

[Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Ubuntu. [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/compare-versions) specifically has been reported by several users to be a seamless experience.

Follow [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install) to setup the WSL and install the `Ubuntu-24.04` distribution. Once your Ubuntu WSL environment is setup, follow the Linux setup instructions to finish setting up your environment. See [these instructions](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for running GUI apps.

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
