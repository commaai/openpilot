# termux-elf-cleaner
Utility for Android ELF files to remove unused parts that the linker warns about.

## Description
When loading ELF files, the Android linker warns about unsupported dynamic section entries with warnings such as:

    WARNING: linker: /data/data/org.kost.nmap.android.networkmapper/bin/nmap: unused DT entry: type 0x6ffffffe arg 0x8a7d4
    WARNING: linker: /data/data/org.kost.nmap.android.networkmapper/bin/nmap: unused DT entry: type 0x6fffffff arg 0x3

This utility strips away the following dynamic section entries:

- `DT_RPATH` - not supported in any Android version.
- `DT_RUNPATH` - supported from Android 7.0.
- `DT_VERDEF` - supported from Android 6.0.
- `DT_VERDEFNUM` - supported from Android 6.0.
- `DT_VERNEEDED` - supported from Android 6.0.
- `DT_VERNEEDNUM` - supported from Android 6.0.
- `DT_VERSYM` - supported from Android 6.0.

It also removes the three ELF sections of type:

- `SHT_GNU_verdef`
- `SHT_GNU_verneed`
- `SHT_GNU_versym`

## Usage
```sh
usage: termux-elf-cleaner <filenames>

Processes ELF files to remove unsupported section types and
dynamic section entries which the Android linker warns about.
```

## Author
Fredrik Fornwall ([@fornwall](https://github.com/fornwall)).

## License

SPDX-License-Identifier: [GPL-3.0-or-later](https://spdx.org/licenses/GPL-3.0-or-later.html)
