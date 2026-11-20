# Navi31 flash tools

Utilities for reading and recovering the 2 MiB SPI flash on Navi31 boards.
Run them from the tinygrad repository root. No image is bundled; keep a verified
full-ROM backup before performing any write.

`fw_live.py` accesses BAR5 through tinygrad's `PCIDevice.map_bar()` abstraction
and supports either the custom ASM24 USB-PCIe bridge or native PCIe. Select the
transport before the subcommand:

```sh
python3 extra/amdflash/fw_live.py --transport usb probe
python3 extra/amdflash/fw_live.py --transport pci probe
```

The default, `--transport auto`, considers USB devices first and then native
PCI devices. Native PCI access requires the usual tinygrad PCI permissions and
an unbound kernel driver.

## Access paths and hardware state

The paths are state-dependent and are not interchangeable:

* **`romless.py`** drives SMUIO `ROM_SW_*` directly through the ASM24 bridge.
  Use it only when an empty or corrupt flash has stalled the PSP PBL. Healthy
  autonomous boot gates this engine; the usual gated status is
  `ROM_SW_STATUS=0x04000800`.
* **`fw_live.py probe`** queries the early PSP boot-firmware mailbox.
* Firmware-mediated write commands are retained for protocol documentation but
  are disabled because an exact stock reflash did not validate safely.
* **`fw_live.py dump`** reads an exact 2 MiB raw image through
  `ROM_INDEX/ROM_DATA`. It refuses devices where the raw SMUIO controller is
  unavailable; the NBIO SOC15 function-ROM aperture is not a physical SPI
  mapping and is deliberately not used as a fallback.

The tools do not reset or power-cycle the board.

## Raw ROM_SW recovery

Identification and read-only operations:

```sh
python3 extra/amdflash/romless.py info
python3 extra/amdflash/romless.py read 0 0x40
python3 extra/amdflash/romless.py dump spi.bin
python3 extra/amdflash/romless.py verify known-good.bin
```

Restore an exact 2 MiB image:

```sh
python3 extra/amdflash/romless.py flash known-good.bin --yes
```

If GD25 status-register bit `SR2.CMP` protects the complete array, clearing it
requires separate authorization:

```sh
python3 extra/amdflash/romless.py flash known-good.bin --clear-cmp --yes
```

Programming is sector-granular. Every written 4 KiB sector is immediately read
back and compared with the input. A range can be resumed independently:

```sh
python3 extra/amdflash/romless.py flash known-good.bin \
  --start-sector 128 --sector-count 64 --yes
```

Navi31 ROM_SW details used by the implementation:

* `ROM_SW_COMMAND = (address << 8) | opcode`
* TX data uses big-endian stream dwords
* `RETURN_DATA_EN` (bit 19) is clear for TX and set for RX
* the RX window exposes the preceding transaction, so reads are primed once

## Firmware-mediated access

The read-only commands are:

```sh
python3 extra/amdflash/fw_live.py probe
python3 extra/amdflash/fw_live.py dump current-spi.bin
```

`dump` produces exactly `0x200000` bytes, requires the raw IFWI magic at offset
zero, rejects mirrored 1 MiB apertures, and restores the ROM controller/index
state before writing output.

The validated early-firmware sequence is available as:

```sh
python3 extra/amdflash/fw_live.py --transport usb ifwi-all full-ifwi.bin --yes
```

It resolves at most Navi31's configured 19 items, streams the item associated
with terminal phase `0x2xx`, and then stops. PSP selects the destination
partition; item `0x08` always comes from the payload referenced by the first
ISH descriptor, matching AMDVBFlash. A hard power cycle is required afterward.

A successful PSP update is not a byte-identical raw rewrite. On the validated
stock test, both A/B payloads matched the source exactly, PSP selected and
booted the updated B partition, and firmware changed only its update cookie,
B descriptor counter/checksum, and generated metadata near `0x1ef000`.

The `stream`, `ifwi-step`, and `live-flash` commands remain disabled. Testing
showed that the PSP live path parses a raw stock IFWI but fails with status
`0xC` (`PSP Write To SPI Error`) after writing an `$AMDVBFL` cookie. Use the
verified ROM_SW path for recovery.

## Safety

ROM_SW erase/program and `ifwi-all` commands require `--yes`; other
firmware-streaming commands are disabled. Read-only commands still touch controller and mailbox registers but
do not issue SPI program/erase or PSP transfer-start commands. Preserve a
known-good full dump outside the repository.
