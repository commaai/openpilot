# beampilot
**beampilot is a fork of openpilot with a bridge to connect BeamNG.tech.**

![Hackatime Badge](https://hackatime-badge.hackclub.com/U0ALZ4JLUL9/beampilot)

<table>
  <tr>
    <td>Currently inoperative</td>
  </tr>
</table>

## Using Beampilot
### You need:
1. a supported device; see [device support](device-support)
2. software; (this repo)

## Device support
As of August 2026, there are two primary performance tiers of comma models: standard and chestnut.<br>
Chestnut class models will take significantly more compute power; however, it will have better driving performance.<br>
**Requirements here only include beampilot usage, and does not include BeamNG's requirements.**

### Device Requirements
* Linux; preferably Ubuntu; tested on Arch
* a GPU that supports [tinygrad](https://tinygrad.org])
* tinygrad gpu backend driver
* any modern desktop CPU
* 4GB VRAM for standard, 8GB VRAM for chestnut class models
* 16GB DRAM for standard, 32GB DRAM for chestnut class models

*(these limits are pretty conservative and are not tested hard limits)*

## Setup
### Settings
Settings/configs can be changed inside `config_beampilot.sh`:
* `FINGERPRINT`
  - by default set to `HONDA_CIVIC_2022`
  - changing this will cause issues with `beamngd` and `beamcamd`;<br>camera position values and CAN data order/structures are different per car
  - you should only change this if you know what you're doing
* `SKIP_FW_QUERY`
  - by default set to `1`
  - this skips the car fingerprinting check.
  - always should be at `1` as without skip openpilot will not boot fully.
* `USE_AMD` or `USE_NV`
  - by default set `USE_AMD`=`1`
  - set only one to `1`, for your GPU type
* `BIG`
  - changes between comma 3/3x display (big) or 4 display (small)
  - codenamed tici (big) or mici (small) in openpilot
* `CHESTNUT`
  - changes between chestnut class (eGPU/dGPU) models or standard (mobile)
  - chestnut class models require 8GB+ VRAM

### Setup Script
```bash
setup_beampilot.sh
```
It will source `config_beampilot.sh` for settings.

### Launch Script
```bash
launch_beampilot.sh
```
It will source `config_beampilot.sh` for settings.

## Model Selection
There are two primary types of models in openpilot from comma: standard and chestnut. See more [here](https://blog.comma.ai/chestnut).<br>
By default, stock openpilot detects Chestnut connected to comma hardware.<br>
In beampilot, chestnut class models can be enabled using the environ `CHESTNUT` set to `1`:<br>
```bash
export CHESTNUT="1"
```

## Processes Changes
### Removed
Processes that are incompatible with desktop use or has no use for in desktop simulation have been removed.
* `camerad` (comma hardware cameras, replaced with `beamcamd`)
* `webcamerad` (unused webcam alternative, replaced with `beamcamd`)
* `micd` (comma hardware mics, removed as not required)
* `dmonitoringmodeld` and `dmonitoringd` (driver monitor, not required for simulation)
  - **It is dangerous and against comma requirements to disable driver monitoring in real vehicles.**
  - **Beampilot is for simulation use ONLY.**
* `sensord` (comma physics data, replaced by `beamngd`)
* `pandad` (comma panda CAN data, replaced by `beamngd`)
* `_pandad` (`pandad` backup, replaced by `beamngd`)
* `updated` (comma hardware OS updater, removed as not required)
* `qcomgpsd` (comma hardware GPS, removed as not required)
* `ubloxd` (comma hardware GPS, removed as not required)
* `pigeond` (comma hardware GPS, removed as not required)
* `modem` (comma hardware modem for cellular, removed as not required)

### Added
Proccesses that are required because of the removed processes have been added.
* `beamngd` Updates Telemetry (100Hz) (currently inop)
* `beamcamd` Updates Cameras (20Hz) (currently inop)

## Test Scripts
venv `source .venv/bin/activate`<br>
modeld `OCL_ICD_VENDORS=amdocl64.icd HIP_VISIBLE_DEVICES=0 openpilot/selfdrive/modeld/modeld.py`<br>
replay `openpilot/tools/replay/replay --demo -b modelV2,drivingModelData,cameraOdometry`<br>
ui `BIG="1" openpilot/selfdrive/ui/ui.py`