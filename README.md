# beampilot

**beampilot is a fork of openpilot with a bridge to connect BeamNG.tech.**

<table>
  <tr>
    <td>Demo in progress</td>
  </tr>
</table>


## Using Beampilot
### Requirements
1. a supported device; see [device support](device-support)
2. software; (this repo)

## Device support
As of August 2026, there are two primary performance tiers of comma models: standard and chestnut.
Chestnut class models will take significantly more compute power; however, it will have better driving performance.

**Requirements here only include beampilot usage, and does not include BeamNG's requirements.**

### Universal Requirements
* a GPU that supports [tinygrad](tinygrad.org)
* tinygrad gpu backend driver
* Any 6-core desktop CPU
* Linux
* and the following:

### Standard
* 4GB VRAM
* 16GB DRAM

### Requirements - Chestnut
* 8GB VRAM
* 32GB DRAM

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
* `beamngd` Updates Telemetry (100Hz)
* `beamcamd` Updates Cameras (20Hz)